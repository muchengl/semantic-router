import sys
from pathlib import Path

import pytest
import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402


def _render_envoy_config(
    tmp_path, monkeypatch, config_text, *, extproc_host, router_api_host
):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "envoy.yaml"
    config_path.write_text(config_text)

    monkeypatch.setenv("ENVOY_EXTPROC_ADDRESS", extproc_host)
    monkeypatch.setenv("ENVOY_ROUTER_API_ADDRESS", router_api_host)

    user_config = parse_user_config(str(config_path))
    generate_envoy_config_from_user_config(user_config, str(output_path))
    return yaml.safe_load(output_path.read_text())


def _cluster_by_name(rendered_config, cluster_name):
    for cluster in rendered_config["static_resources"]["clusters"]:
        if cluster["name"] == cluster_name:
            return cluster
    raise AssertionError(f"cluster {cluster_name!r} not found")


def test_generate_envoy_config_uses_logical_dns_for_split_extproc_host(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "test-model"
  models:
    - name: "test-model"
      backend_refs:
        - name: "primary"
          endpoint: "host.docker.internal:8000"
          protocol: "http"
          weight: 100
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="vllm-sr-router-container",
        router_api_host="vllm-sr-router-container",
    )

    cluster = _cluster_by_name(rendered, "extproc_service")

    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["dns_lookup_family"] == "V4_ONLY"
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert (
        endpoint["address"]["socket_address"]["address"] == "vllm-sr-router-container"
    )
    assert endpoint["hostname"] == "vllm-sr-router-container"


def _model_route(rendered_config, model_name):
    """Find the route entry whose x-selected-model header matches *model_name*."""
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    routes = hcm["route_config"]["virtual_hosts"][0]["routes"]
    for route in routes:
        headers = route.get("match", {}).get("headers", [])
        for h in headers:
            if (
                h.get("name") == "x-selected-model"
                and h.get("string_match", {}).get("exact") == model_name
            ):
                return route
    raise AssertionError(f"route for model {model_name!r} not found")


def _path_rewrite_lua(rendered_config):
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    for http_filter in hcm["http_filters"]:
        typed_config = http_filter.get("typed_config", {})
        inline_code = typed_config.get("inline_code", "")
        if "SELECTED_ROUTE_BY_MODEL" in inline_code:
            return http_filter, inline_code
    raise AssertionError("path rewrite Lua filter not found")


def _lua_route_spec(lua_code, model_name):
    marker = f'["{model_name}"] = {{'
    start = lua_code.index(marker)
    end = lua_code.index("},", start) + 2
    return lua_code[start:end]


def _original_path_header_mutation(rendered_config):
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    for http_filter in hcm["http_filters"]:
        if http_filter["name"] == "envoy.filters.http.header_mutation":
            return http_filter
    raise AssertionError("original-path header mutation filter not found")


def _default_route(rendered_config):
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    return hcm["route_config"]["virtual_hosts"][0]["routes"][-1]


def _assert_original_path_pipeline(rendered_config, lua_filter):
    listener = rendered_config["static_resources"]["listeners"][0]
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    http_filters = hcm["http_filters"]
    header_mutation = _original_path_header_mutation(rendered_config)
    request_mutation = header_mutation["typed_config"]["mutations"][
        "request_mutations"
    ][0]["append"]
    assert request_mutation["header"] == {
        "key": "x-vsr-original-path",
        "value": "%REQ(:PATH)%",
    }
    assert request_mutation["append_action"] == "OVERWRITE_IF_EXISTS_OR_ADD"

    header_mutation_index = http_filters.index(header_mutation)
    extproc_index = next(
        index
        for index, item in enumerate(http_filters)
        if item["name"] == "envoy.filters.http.ext_proc"
    )
    extproc_config = http_filters[extproc_index]["typed_config"]
    assert extproc_config["mutation_rules"] == {
        "disallow_expression": {
            "regex": "^x-vsr-original-path$",
        },
        "disallow_is_error": True,
    }
    lua_index = http_filters.index(lua_filter)
    router_index = next(
        index
        for index, item in enumerate(http_filters)
        if item["name"] == "envoy.filters.http.router"
    )
    assert header_mutation_index < extproc_index < lua_index < router_index


def test_path_rewrite_lua_runs_after_model_selection(tmp_path, monkeypatch):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "gemini-model"
  models:
    - name: "gemini-model"
      provider_model_id: "gemini-model"
      backend_refs:
        - name: "gemini"
          base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
          provider: "openai"
          weight: 100
    - name: "v1-proxy-model"
      provider_model_id: "v1-proxy-model"
      backend_refs:
        - name: "proxy"
          base_url: "https://api.example.com/v1/proxy"
          provider: "openai"
          weight: 100
routing:
  modelCards:
    - name: "gemini-model"
    - name: "v1-proxy-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "gemini-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    for route in (
        _model_route(rendered, "gemini-model"),
        _model_route(rendered, "v1-proxy-model"),
        _default_route(rendered),
    ):
        assert "regex_rewrite" not in route["route"]

    lua_filter, lua_code = _path_rewrite_lua(rendered)
    gemini_route = _lua_route_spec(lua_code, "gemini-model")
    assert 'path_prefix = "/v1beta/openai"' in gemini_route
    assert 'chat_path = "/v1beta/openai/chat/completions"' in gemini_route
    proxy_route = _lua_route_spec(lua_code, "v1-proxy-model")
    assert 'path_prefix = "/v1/proxy"' in proxy_route
    assert 'chat_path = "/v1/proxy/chat/completions"' in proxy_route
    assert 'local DEFAULT_MODEL = "gemini-model"' in lua_code
    assert "request_handle:body()" in lua_code
    assert 'headers:get("x-vsr-original-path")' in lua_code
    assert 'headers:get(":path")' not in lua_code
    assert "rewrite_path(original_path, selected_route)" in lua_code
    assert 'if path == "/v1/responses" then' in lua_code
    assert "missing immutable x-vsr-original-path header" in lua_code
    assert '[":status"] = "500"' in lua_code
    assert "x-envoy-original-path" not in lua_code
    _assert_original_path_pipeline(rendered, lua_filter)


def test_path_rewrite_route_carries_provider_chat_path(tmp_path, monkeypatch):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: custom-model
  models:
    - name: custom-model
      backend_refs:
        - name: custom
          base_url: https://api.example.com/v1
          provider: openai
          chat_path: /custom/chat
    - name: azure-model
      backend_refs:
        - name: azure
          base_url: https://example.openai.azure.com/openai/deployments/gpt-4o
          provider: azure-openai
          api_version: "2024-10-21"
routing:
  modelCards:
    - name: custom-model
    - name: azure-model
  decisions:
    - name: default-route
      description: default route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: custom-model
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    _, lua_code = _path_rewrite_lua(rendered)
    custom_route = _lua_route_spec(lua_code, "custom-model")
    assert 'path_prefix = "/v1"' in custom_route
    assert 'chat_path = "/custom/chat"' in custom_route

    azure_route = _lua_route_spec(lua_code, "azure-model")
    assert 'path_prefix = "/openai/deployments/gpt-4o"' in azure_route
    assert (
        'chat_path = "/openai/deployments/gpt-4o/chat/completions'
        '?api-version=2024-10-21"'
    ) in azure_route

    responses_branch = lua_code.index('if path == "/v1/responses" then')
    selected_prefix_branch = lua_code.index(
        "if path_has_prefix(path, selected_route.path_prefix) then"
    )
    assert responses_branch < selected_prefix_branch
    assert 'if route_query ~= "" then' in lua_code


def test_backend_ref_ip_port_path_produces_correct_envoy_cluster_and_route(
    tmp_path, monkeypatch
):
    """Backend ref http://10.0.0.1:8000/v1 should split into address=10.0.0.1,
    port=8000, host_authority=10.0.0.1:8000, path_prefix=/v1, and the route
    should leave path rewriting to the post-ext_proc Lua filter."""
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "test-model"
  models:
    - name: "test-model"
      backend_refs:
        - name: "primary"
          endpoint: "http://10.0.0.1:8000/v1"
          weight: 100
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    # --- cluster assertions ---
    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["connect_timeout"] == "10s"
    assert cluster["type"] == "STATIC"
    ep = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert ep["address"]["socket_address"]["address"] == "10.0.0.1"
    assert ep["address"]["socket_address"]["port_value"] == 8000

    # --- route assertions ---
    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    assert route_action["host_rewrite_literal"] == "10.0.0.1:8000"
    assert "regex_rewrite" not in route_action
    _, lua_code = _path_rewrite_lua(rendered)
    route_spec = _lua_route_spec(lua_code, "test-model")
    assert 'path_prefix = "/v1"' in route_spec
    assert "chat_path" not in route_spec


def test_provider_reliability_renders_retry_outlier_and_least_request(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: test-model
  models:
    - name: test-model
      reliability:
        lb_policy: least_request
        retry_count: 2
        retry_on: connect-failure,refused-stream
        consecutive_5xx: 5
        base_ejection_time: 45s
        max_ejection_percent: 25
        health_check_path: /health
        health_check_interval: 15s
        health_check_timeout: 3s
      backend_refs:
        - endpoint: 10.0.0.1:8000
        - endpoint: 10.0.0.2:8000
routing:
  modelCards:
    - name: test-model
  decisions:
    - name: default-route
      description: default route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: test-model
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    route = _model_route(rendered, "test-model")["route"]
    assert route["retry_policy"] == {
        "retry_on": "connect-failure,refused-stream",
        "num_retries": 2,
    }
    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["lb_policy"] == "LEAST_REQUEST"
    assert cluster["least_request_lb_config"]["choice_count"] == 2
    assert cluster["outlier_detection"]["consecutive_5xx"] == 5
    assert cluster["outlier_detection"]["base_ejection_time"] == "45s"
    assert cluster["outlier_detection"]["max_ejection_percent"] == 25
    assert cluster["health_checks"][0]["http_health_check"]["path"] == "/health"
    assert cluster["health_checks"][0]["interval"] == "15s"
    assert cluster["health_checks"][0]["timeout"] == "3s"
    assert cluster["circuit_breakers"]["thresholds"][0]["max_requests"] == 4096


def test_backend_ref_domain_with_path_produces_correct_envoy_cluster_and_route(
    tmp_path, monkeypatch
):
    """Backend ref https://api.example.com/compatible-mode/v1 should produce
    address=api.example.com, port=443, host_authority=api.example.com (standard
    port omitted), LOGICAL_DNS cluster, and Lua path-prefix rewriting."""
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "test-model"
  models:
    - name: "test-model"
      backend_refs:
        - name: "primary"
          endpoint: "https://api.example.com/compatible-mode/v1/"
          weight: 100
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    # --- cluster assertions ---
    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["dns_lookup_family"] == "V4_ONLY"
    ep = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert ep["address"]["socket_address"]["address"] == "api.example.com"
    assert ep["address"]["socket_address"]["port_value"] == 443
    assert ep["hostname"] == "api.example.com"

    # --- route assertions ---
    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    # standard port 443 → host_authority should omit port
    assert route_action["host_rewrite_literal"] == "api.example.com"
    assert "regex_rewrite" not in route_action
    _, lua_code = _path_rewrite_lua(rendered)
    route_spec = _lua_route_spec(lua_code, "test-model")
    assert 'path_prefix = "/compatible-mode/v1"' in route_spec
    assert "chat_path" not in route_spec


def test_backend_ref_https_base_url_uses_tls_and_explicit_extra_headers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "test-model"
  models:
    - name: "test-model"
      provider_model_id: "openai/gpt-4o-mini"
      backend_refs:
        - name: "openrouter"
          base_url: "https://openrouter.ai/api/v1"
          provider: "openai"
          auth_header: "Authorization"
          auth_prefix: "Bearer"
          api_key_env: "OPENROUTER_API_KEY"
          extra_headers:
            X-Test-Trace: "router-flow"
            X-Test-Tenant: "eval"
          weight: 1
routing:
  modelCards:
    - name: "test-model"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "test-model"
          use_reasoning: false
""",
        extproc_host="localhost",
        router_api_host="localhost",
    )

    cluster = _cluster_by_name(rendered, "test_model_cluster")
    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["transport_socket"]["name"] == "envoy.transport_sockets.tls"

    route = _model_route(rendered, "test-model")
    route_action = route["route"]
    assert route_action["host_rewrite_literal"] == "openrouter.ai"
    assert "regex_rewrite" not in route_action
    _, lua_code = _path_rewrite_lua(rendered)
    route_spec = _lua_route_spec(lua_code, "test-model")
    assert 'path_prefix = "/api/v1"' in route_spec
    assert 'chat_path = "/api/v1/chat/completions"' in route_spec

    headers = {
        item["header"]["key"]: item["header"]["value"]
        for item in route["request_headers_to_add"]
    }
    assert headers["Authorization"] == "Bearer sk-test-openrouter"
    assert headers["X-Test-Trace"] == "router-flow"
    assert headers["X-Test-Tenant"] == "eval"


def test_generate_envoy_config_custom_anthropic_upstream_rewrites_host(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "claude-sonnet-4.6"
  models:
    - name: "claude-sonnet-4.6"
      api_format: "anthropic"
      backend_refs:
        - name: "anthropic-primary"
          endpoint: "domain.com:443"
          protocol: "https"
          weight: 100
          base_url: "https://domain.com/Anthropic"
          type: "anthropic"
          provider: "anthropic"
routing:
  modelCards:
    - name: "claude-sonnet-4.6"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "claude-sonnet-4.6"
          use_reasoning: false
""",
        extproc_host="vllm-sr-router-container",
        router_api_host="vllm-sr-router-container",
    )

    route = _model_route(rendered, "claude-sonnet-4.6")
    assert route["route"]["cluster"] == "claude_sonnet_4.6_cluster"
    assert route["route"]["host_rewrite_literal"] == "domain.com"

    with pytest.raises(AssertionError):
        _cluster_by_name(rendered, "anthropic_api_cluster")


def test_generate_envoy_config_uses_logical_dns_for_api_only_router_fallback(
    tmp_path, monkeypatch
):
    rendered = _render_envoy_config(
        tmp_path,
        monkeypatch,
        """
version: v0.3
listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
providers:
  defaults:
    default_model: "claude-test"
  models:
    - name: "claude-test"
      api_format: "anthropic"
routing:
  modelCards:
    - name: "claude-test"
  decisions:
    - name: "default-route"
      description: "default route"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "claude-test"
          use_reasoning: false
""",
        extproc_host="vllm-sr-router-container",
        router_api_host="vllm-sr-router-container",
    )

    cluster = _cluster_by_name(rendered, "vllm_static_cluster")

    assert cluster["type"] == "LOGICAL_DNS"
    assert cluster["dns_lookup_family"] == "V4_ONLY"
    endpoint = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
    assert (
        endpoint["address"]["socket_address"]["address"] == "vllm-sr-router-container"
    )
    assert endpoint["hostname"] == "vllm-sr-router-container"

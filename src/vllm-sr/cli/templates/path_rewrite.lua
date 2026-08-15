local SELECTED_ROUTE_BY_MODEL = {
{% for model in path_rewrite_models %}
  [{{ model.name | tojson }}] = {
    path_prefix = {{ model.path_prefix | tojson }},
    {% if model.chat_path %}
    chat_path = {{ model.chat_path | tojson }},
    {% endif %}
  },
{% endfor %}
}
local DEFAULT_MODEL = {{ models[0].name | tojson }}

local function split_path_and_query(value)
  local query_start = string.find(value, "?", 1, true)
  if query_start == nil then
    return value, ""
  end
  return string.sub(value, 1, query_start - 1), string.sub(value, query_start)
end

local function v1_suffix(path)
  if path == "/v1" then
    return ""
  end
  if string.sub(path, 1, 4) == "/v1/" then
    return string.sub(path, 4)
  end
  return nil
end

local function path_has_prefix(path, prefix)
  if path == prefix then
    return true
  end
  return string.sub(path, 1, string.len(prefix) + 1) == prefix .. "/"
end

local function join_path(prefix, suffix)
  if suffix == "" then
    return prefix
  end
  local prefix_has_slash = string.sub(prefix, -1) == "/"
  local suffix_has_slash = string.sub(suffix, 1, 1) == "/"
  if prefix_has_slash and suffix_has_slash then
    return string.sub(prefix, 1, -2) .. suffix
  end
  if not prefix_has_slash and not suffix_has_slash then
    return prefix .. "/" .. suffix
  end
  return prefix .. suffix
end

local function route_chat_path(selected_route, original_query)
  local chat_path =
    selected_route.chat_path or join_path(selected_route.path_prefix, "/chat/completions")
  local _, route_query = split_path_and_query(chat_path)
  if route_query ~= "" then
    return chat_path
  end
  return chat_path .. original_query
end

local function rewrite_path(original_path, selected_route)
  local path, query = split_path_and_query(original_path)
  if path == "/v1/responses" then
    return route_chat_path(selected_route, query)
  end
  if path == "/v1/chat/completions" and selected_route.chat_path ~= nil then
    return route_chat_path(selected_route, query)
  end
  if path_has_prefix(path, selected_route.path_prefix) then
    return path .. query
  end
  local suffix = v1_suffix(path)
  if suffix == nil then
    return original_path
  end
  return join_path(selected_route.path_prefix, suffix) .. query
end

function envoy_on_request(request_handle)
  local headers = request_handle:headers()

  -- Model selection happens in ext_proc's request-body response.
  if headers:get("x-selected-model") == nil then
    request_handle:body()
  end

  local model = headers:get("x-selected-model") or DEFAULT_MODEL
  local selected_route = SELECTED_ROUTE_BY_MODEL[model]
  if selected_route == nil then
    return
  end

  -- Never derive a rewrite from :path because ext_proc may mutate it.
  local original_path = headers:get("x-vsr-original-path")
  if original_path == nil then
    request_handle:logErr("missing immutable x-vsr-original-path header")
    request_handle:respond(
      {
        [":status"] = "500",
        ["content-type"] = "application/json",
      },
      "{\"error\":{\"message\":\"Internal path rewrite error\"}}"
    )
    return
  end

  headers:replace(":path", rewrite_path(original_path, selected_route))
end

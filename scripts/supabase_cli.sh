#!/usr/bin/env bash

# Resolve Supabase's legacy shim to its co-located Go CLI before deployment.
# This file is sourced so SUPABASE_GO_BINARY remains exported for later calls.
configure_supabase_cli() {
  local supabase_bin candidate shim_version go_version

  supabase_bin="$(command -v supabase 2>/dev/null || true)"
  if [[ -z "$supabase_bin" ]]; then
    echo "Supabase CLI 未安裝，停止部署" >&2
    return 1
  fi

  candidate="${SUPABASE_GO_BINARY:-}"
  if [[ -n "$candidate" ]]; then
    if [[ ! -x "$candidate" ]]; then
      echo "SUPABASE_GO_BINARY 不存在或不可執行：$candidate" >&2
      return 1
    fi
  else
    local search_paths=(
      "$(dirname "$supabase_bin")/supabase-go"
      "$HOME/.local/share/supabase/supabase-go"
    )
    for candidate in "${search_paths[@]}"; do
      if [[ -x "$candidate" ]]; then
        export SUPABASE_GO_BINARY="$candidate"
        break
      fi
    done
    candidate="${SUPABASE_GO_BINARY:-}"
  fi

  if [[ -n "$candidate" ]]; then
    shim_version="$(supabase --version 2>/dev/null | tail -n 1 | tr -d '[:space:]')"
    go_version="$("$candidate" --version 2>/dev/null | tail -n 1 | tr -d '[:space:]')"
    if [[ -z "$shim_version" || "$shim_version" != "$go_version" ]]; then
      echo "Supabase shim/Go CLI 版本不一致：shim=${shim_version:-unknown}, go=${go_version:-unknown}" >&2
      return 1
    fi
    return 0
  fi

  # A standalone Go CLI needs no companion binary. The JS legacy shim does.
  if grep -aFq 'supabase/legacy/LegacyGoProxy' "$supabase_bin" 2>/dev/null; then
    echo "Supabase shim 找不到 supabase-go；請重新安裝完整 CLI" >&2
    return 1
  fi
}

(function () {
  const SOURCE = "ai-poker-wizard-gtow-session";
  const MESSAGE_TYPE = "GTOW_SESSION_OBSERVED";
  const REFRESH_PATH = "/v1/token/refresh/";
  const originalFetch = window.fetch;
  const originalXhrOpen = XMLHttpRequest.prototype.open;
  const originalXhrSetRequestHeader = XMLHttpRequest.prototype.setRequestHeader;
  const originalXhrSend = XMLHttpRequest.prototype.send;

  function headerValue(headers, name) {
    if (!headers) return "";
    if (typeof Headers !== "undefined" && headers instanceof Headers) {
      return headers.get(name) || "";
    }
    if (Array.isArray(headers)) {
      const found = headers.find(([key]) => key?.toLowerCase() === name.toLowerCase());
      return found ? found[1] : "";
    }
    const direct = headers[name] || headers[name.toLowerCase()] || headers[name.toUpperCase()];
    return direct || "";
  }

  function bearerFromHeaders(headers) {
    const authorization = headerValue(headers, "authorization");
    const match = typeof authorization === "string" ? authorization.match(/^Bearer\s+(.+)$/i) : null;
    return match ? match[1] : "";
  }

  function clientIdFromHeaders(headers) {
    return headerValue(headers, "gwclientid") || headerValue(headers, "GWCLIENTID");
  }

  function urlFromInput(input) {
    if (typeof input === "string") return input;
    if (input instanceof URL) return input.href;
    return input?.url || "";
  }

  function headersFromInput(input, init) {
    const headers = new Headers();
    const appendAll = (source) => {
      if (!source) return;
      if (typeof Headers !== "undefined" && source instanceof Headers) {
        source.forEach((value, key) => headers.set(key, value));
        return;
      }
      if (Array.isArray(source)) {
        source.forEach(([key, value]) => headers.set(key, value));
        return;
      }
      Object.entries(source).forEach(([key, value]) => headers.set(key, value));
    };
    appendAll(input?.headers);
    appendAll(init?.headers);
    return headers;
  }

  function gtowApiUrl(value) {
    try {
      const url = new URL(value, window.location.origin);
      return url.origin === "https://api.gtowizard.com" ? url : null;
    } catch {
      return null;
    }
  }

  function postObserved(accessToken, clientId) {
    if (!accessToken) return;
    window.postMessage({
      source: SOURCE,
      type: MESSAGE_TYPE,
      accessToken,
      clientId: clientId || "",
      observedAt: new Date().toISOString(),
    }, window.location.origin);
  }

  function accessFromRefreshBody(body) {
    if (!body || typeof body !== "object") return "";
    return body.access || body.access_token || body.accessToken || "";
  }

  window.fetch = function wrappedFetch(input, init) {
    const url = urlFromInput(input);
    const headers = headersFromInput(input, init);
    const clientId = clientIdFromHeaders(headers);
    const gtowUrl = gtowApiUrl(url);
    if (gtowUrl) postObserved(bearerFromHeaders(headers), clientId);

    const responsePromise = originalFetch.apply(this, arguments);
    if (gtowUrl?.pathname === REFRESH_PATH) {
      Promise.resolve(responsePromise).then((response) => {
        response.clone().json().then((body) => {
          postObserved(accessFromRefreshBody(body), clientId);
        }).catch(() => {});
      }).catch(() => {});
    }
    return responsePromise;
  };

  XMLHttpRequest.prototype.open = function wrappedOpen(method, url) {
    this.__apwGtowUrl = String(url || "");
    this.__apwGtowHeaders = {};
    return originalXhrOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.setRequestHeader = function wrappedSetRequestHeader(name, value) {
    if (this.__apwGtowHeaders) this.__apwGtowHeaders[String(name).toLowerCase()] = String(value);
    return originalXhrSetRequestHeader.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function wrappedSend() {
    const headers = this.__apwGtowHeaders || {};
    const clientId = clientIdFromHeaders(headers);
    const gtowUrl = gtowApiUrl(this.__apwGtowUrl || "");
    if (gtowUrl) postObserved(bearerFromHeaders(headers), clientId);
    if (gtowUrl?.pathname === REFRESH_PATH) {
      this.addEventListener("load", () => {
        try {
          postObserved(accessFromRefreshBody(JSON.parse(this.responseText)), clientId);
        } catch {}
      });
    }
    return originalXhrSend.apply(this, arguments);
  };
})();

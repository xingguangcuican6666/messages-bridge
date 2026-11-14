package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
)

// newPassthroughProxy forwards every path this bridge does not translate to the
// upstream unchanged, including request and response bodies and streaming.
func newPassthroughProxy(cfg config) (*httputil.ReverseProxy, error) {
	target := *cfg.upstreamBase
	proxy := &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			setTargetURL(request, &target)
			if cfg.upstreamKey != "" {
				request.Out.Header.Set("Authorization", "Bearer "+cfg.upstreamKey)
			}
		},
		// Stream pass-through responses immediately instead of buffering them.
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			log.Printf("passthrough failed: %v", err)
			writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream request failed")
		},
	}
	return proxy, nil
}

func setTargetURL(request *httputil.ProxyRequest, target *url.URL) {
	request.SetURL(target)
	request.Out.Host = target.Host
}

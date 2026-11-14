FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/messages-bridge .

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=build /out/messages-bridge /usr/local/bin/messages-bridge
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/messages-bridge"]

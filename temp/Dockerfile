# 1. Use the standard Intel-based image to do the compiling
FROM --platform=$BUILDPLATFORM gcc:latest AS builder

RUN apt-get update && apt-get install -y nlohmann-json3-dev

WORKDIR /app
COPY hello.cpp .

# 2. Tell G++ specifically to target ARM64 (for your Mac)
# We use -static to make sure it doesn't need external libraries
RUN g++ -o json_app hello.cpp -static

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/json_app .
CMD ["./json_app"]
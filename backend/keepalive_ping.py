import os
import sys
import urllib.error
import urllib.request

HEALTHCHECK_URL = os.getenv(
    "HEALTHCHECK_URL",
    "https://plant-disease-detector-tmla.onrender.com/health",
)


def main() -> int:
    request = urllib.request.Request(HEALTHCHECK_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.getcode()
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        print(f"Ping failed with HTTP status {exc.code}: {HEALTHCHECK_URL}")
        return 1
    except urllib.error.URLError as exc:
        print(f"Ping failed: {exc.reason}")
        return 1

    if status >= 400:
        print(f"Ping returned non-success status {status}: {HEALTHCHECK_URL}")
        return 1

    print(f"Ping success ({status}) -> {HEALTHCHECK_URL} | response: {body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

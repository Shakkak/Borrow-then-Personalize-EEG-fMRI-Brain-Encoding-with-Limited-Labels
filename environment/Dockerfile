# CPU image for the tests and the figure scripts (no GPU, no datasets). From the repository root:
#   docker build -t xsub-torch:local -f environment/Dockerfile .
#   docker run --rm -v "$PWD":/repo -w /repo xsub-torch:local sh -c 'pip install -q --no-deps -e . && python -m pytest -q'
FROM python:3.12-slim
COPY environment/requirements-tested.txt /tmp/requirements-tested.txt
RUN grep -v -E '^(torch|nilearn|nibabel|lxml|requests|certifi|charset-normalizer|idna|urllib3)==' /tmp/requirements-tested.txt > /tmp/base.txt \
 && pip install -q --root-user-action=ignore --no-cache-dir -r /tmp/base.txt \
 && pip install -q --root-user-action=ignore --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
CMD ["python3"]

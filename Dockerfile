# Phase 6: containerize app/demo.py for deploy to a brendenrunion.com subdomain.
FROM python:3.12-slim

WORKDIR /app

# Copy only what pip install -e needs first, so dependency install layers
# cache independently of source edits.
COPY pyproject.toml README.md ./
COPY misinfo_agent ./misinfo_agent
# Editable install: misinfo_agent/data/source_credibility.csv is read via a
# path relative to __file__ at runtime, not packaged via package-data, so
# tool_assess_source needs the source tree itself on disk (as -e gives it),
# not just the importable modules a regular wheel build would produce.
RUN pip install --no-cache-dir -e ".[agent,demo]"

COPY app ./app

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app/demo.py"]

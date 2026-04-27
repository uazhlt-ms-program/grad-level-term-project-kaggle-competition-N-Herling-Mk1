# Build on top of the course-provided image so grading-environment
# parity is preserved. We add only the Python packages this project needs.
FROM uazhlt/python-playground:latest

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]

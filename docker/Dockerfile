FROM python:3.12

WORKDIR /dynatree

COPY docker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts ./scripts/
RUN mkdir /root/solara_log

EXPOSE 8700
WORKDIR scripts
CMD ["solara", "run", "solara_app.py", "--host=0.0.0.0", "--port=8700"]

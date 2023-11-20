FROM python:3.10.10
COPY . /synsteg
WORKDIR /synsteg
RUN pip install -r requirements.txt
CMD ["python", "synsteg_bot.py"]

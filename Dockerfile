FROM python

WORKDIR /House_price_prediction

COPY . .

CMD ["python", "app.py"]
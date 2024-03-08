import shutil
import tempfile
import xarray as xr
import urllib.request
import pandas as pd

#%% Trying out wth sarracenia
import sarracenia
import sarracenia.moth
import sarracenia.credentials
from sarracenia.config import default_config

import os
import time
import socket

cfg = default_config()
cfg.logLevel = 'debug'
cfg.broker = sarracenia.credentials.Credential('amqp://tfeed:password@localhost')
cfg.exchange = 'xpublic'
cfg.post_baseUrl = 'http://host'
cfg.post_baseDir = '/tmp'

posting_engine = sarracenia.moth.Moth.pubFactory( cfg.dictify() )

# create a file?
sample_fileName = '/tmp/sample.txt'
sample_file = open( sample_fileName , 'w')
sample_file.write(
"""
CACN00 CWAO 161800
PMN
160,2021,228,1800,1065,100,-6999,20.49,43.63,16.87,16.64,323.5,9.32,27.31,1740,317.8,19.22,1.609,230.7,230.7,230.7,230.7,0,0,0,16.38,15.59,305.
9,17.8,16.38,19.35,55.66,15.23,14.59,304,16.67,3.844,20.51,18.16,0,0,-6999,-6999,-6999,-6999,-6999,-6999,-6999,-6999,0,0,0,0,0,0,0,0,0,0,0,0,0,
13.41,13.85,27.07,3473
"""
)
sample_file.close()

# supply msg init the to file
# you can supply msg_init with your files, it will build a message appropriate for it.
m = sarracenia.Message.fromFileData(sample_fileName, cfg, os.stat(sample_fileName) )
# here is the resulting message.
print(m)

# feed the message to the posting engine.
posting_engine.putNewMessage(m)

# when done, should close... cleaner...
posting_engine.close()

#%% Trying out https://www.pathandfocus.com/blog/how-to-get-canadian-forecasting-data (but modified for python)
import pika
import json

AMQP_URL = 'amqps://anonymous:anonymous@dd.weather.gc.ca/?heartbeat=60'
EXCHANGE = 'xpublic'
EXPIRES = 60000  # Set your expiration timeout in milliseconds
QUEUE_NAME = 'YOUR_QUEUE_NAME'  # Replace with your queue name
PYTHON_API = 'YOUR_PYTHON_API'  # Replace with your Python API endpoint or function

def handle_grib2_file(url):
    # Implement your logic to handle the GRIB2 file here.
    print(f"Handling GRIB2 file from URL: {url}")
    # You might want to call your PYTHON_API here or perform some action with the URL.

def start():
    parameters = pika.URLParameters(AMQP_URL)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Declare the queue, if it doesn't exist, it will be created.
    channel.queue_declare(queue=QUEUE_NAME, arguments={'x-expires': EXPIRES})

    # Bind the queue to the exchange.
    channel.queue_bind(queue=QUEUE_NAME, exchange=EXCHANGE, routing_key='v02.post.model_hrdps.continental.#TMP_AGL-2m#')

    print(f'Subscribed to queue: {QUEUE_NAME}')

    def callback(ch, method, properties, body):
        message_content = body.decode('utf-8')
        time, host, pathname = message_content.split(' ')
        url = f"http://{host}{pathname}"

        handle_grib2_file(url)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    # Start consuming messages from the queue.
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)

    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

# if __name__ == '__main__':
#     start()

#%% Updated version with ChatGPT
AMQP_URL = 'amqp://anonymous:anonymous@dd.weather.gc.ca'
EXCHANGE = 'xpublic'
QUEUE_NAME = 'q_anonymous.datamart_extract.IRDA'

def connect_to_amqp():
    parameters = pika.URLParameters(AMQP_URL)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Declare a queue
    channel.queue_declare(queue=QUEUE_NAME, durable=True, arguments={'x-expires': 300000})  # Expiry time in ms

    # Bind the queue to the exchange with your specific routing key
    routing_key = 'v02.post.model_hrdps.continental.#TMP_AGL-2m#'
    channel.queue_bind(exchange=EXCHANGE, queue=QUEUE_NAME, routing_key=routing_key)

    return channel

def callback(ch, method, properties, body):
    # Process the message
    message = json.loads(body)
    print(f"Received: {message}")
    # Implement your data processing or downloading logic here

def start_consuming(channel):
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    print(f"Subscribed to {QUEUE_NAME}. Waiting for messages.")
    channel.start_consuming()

channel = connect_to_amqp()
start_consuming(channel)


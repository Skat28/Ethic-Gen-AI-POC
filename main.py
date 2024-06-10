import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymupdf4llm
# import fitz
import pymupdf
import pymupdf4llm
import pathlib
from langchain.text_splitter import MarkdownTextSplitter
import os
from openai import OpenAI
from pinecone import Pinecone
import textwrap
import pathlib 
from flask import Flask
from flask import Flask, request,redirect, url_for, send_from_directory, render_template
import re
from pathlib import Path
import boto3
import ast
import time
# -----------------------------------------aws code----------------------------------------
while True:
    import boto3

    # Create SQS client
    sqs = boto3.client('sqs')

    queue_url = 'https://sqs.us-east-1.amazonaws.com/239902576304/week7-gen-ai'

    # Receive message from SQS queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=0,
        WaitTimeSeconds=0
    )
    # print(response['Messages'][0]['Body'])
    # print(f"type: {type(response['Messages'][0]['Body'])}")
    str_response = response['Messages'][0]['Body']
    str_dict = ast.literal_eval(str_response)
    # print(f'converted to dict:{str_dict} and type{type(str_dict)}')
    s3_bucket_info = str_dict['Records'][0]['s3']['bucket']
    s3_content_info = str_dict['Records'][0]['s3']['object']
    print('we have received a bucket with the follwoing information:')
    print(f'{s3_bucket_info}')
    print('\n\n bucket contents are:{str_dict}')
    print(f'\n{s3_content_info}')
    # time.sleep(15)

    
    message = s3_bucket_info
    # receipt_handle = message['ReceiptHandle']


    # # Delete received message from queue
    # sqs.delete_message(
    #     QueueUrl=queue_url,
    #     ReceiptHandle=receipt_handle
    # )
    # print(f"Received and deleted message: {type(message['Body'])}")


# ----------------------------aws code ends here------------------------------------------
'''
def extract_tables(md_text):
    tables = table_pattern.findall(md_text)
    return tables

# Function to replace tables with better format
def replace_tables(md_text, combined_tables):
    replaced_text = md_text
    for heading, table in combined_tables.items():
        table_text = f"\n\nIdentifying Table for {heading}:\n\n{table.strip()}\n\n{'-' * 40}\n\n"
        replaced_text = replaced_text.replace(table, table_text)
    return replaced_text
def get_embeddings(text, model = "text-embedding-ada-002"):
  text = text.replace("\n"," ")
  return client.embeddings.create(input = text, model = model).data[0].embedding


def generate_ids(number, size):
  import string, random
  ids=[]
  for i in range(number):
    res = ''.join(random.choices(string.ascii_letters,k=size))
    ids.append(res)
    if len(set(ids)) != i+1:
      i-=1
      ids.pop(-1)

  return ids


# Define a function to load chunks into a DataFrame
def load_chunks(df, split_text):
    ids = generate_ids(len(split_text), 7)
    for i, chunk_text in enumerate(split_text):
      if i<50:
        df.loc[i] = [ids[i], get_embeddings(chunk_text, model='text-embedding-ada-002'), {'text': chunk_text,'document_name':'aws_overview.pdf'}]

def prepare_DF(df):
    import json,ast
    try: df=df.drop('Unnamed: 0',axis=1)
    except: print('Unnamed Not Found')
    df['values']=df['values'].apply(lambda x: np.array([float(i) for i in x.replace("[",'').replace("]",'').split(',')]))
    df['metadata']=df['metadata'].apply(lambda x: ast.literal_eval(x))
    return df

def chunker(seq, size):

    'Yields a series of slices of the original iterable, up to the limit of what size is.'
    for pos in range(0, len(seq), size):
        yield seq.iloc[pos:pos + size]

def convert_data(chunk):
    'Converts a pandas dataframe to be a simple list of tuples, formatted how the `upsert()` method in the Pinecone Python client expects.'
    data = []
    for i in chunk.to_dict('records'):
        data.append(i)
    return data
print(f' directory:{os.getcwd()}')
def to_be_executed(file_name):
    # file_name=os.environ.get('file_name')
    # # Regular expression pattern to identify Markdown tables
    # print(f'we have received the file named:{file_name}')
    table_pattern = re.compile(r'(\|.*?\|\n(\|.*?---.*?\|\n)(\|.*?\|\n)*)')

    # Function to extract tables using the regex pattern
    # Convert the document to markdown or read the markdown
    path_to_new_file=(f'pdf-files/{file_name}.pdf')
    # path_to_new_file=f"pdf-files/{file_name}.pdf"
    # path_to_new_file=str(path_to_new_file)
    
    md_text = pymupdf4llm.to_markdown(path_to_new_file)
    print(f'extracted markdown text from file and saved in md_text variable')
    # Extract tables from the Markdown text
    tables = extract_tables(md_text)
    print(f'extracted tables from md_text')
    # Dictionary to store tables based on their category headings
    combined_tables = {}


      # --------------------------------
    # Combine tables with similar category headings
    for table in tables:
        heading = table[0].split('|')[1].strip()
        if heading in combined_tables:
            combined_tables[heading] += table[0]
        else:
            combined_tables[heading] = table[0]

    # Replace tables with better format
    updated_md_text = replace_tables(md_text, combined_tables)
    print(f'updated markedown text with better formatting for tables')
    # Write the updated Markdown text to the output file
    output_file_path = f"output-files/{file_name}.md"
    pathlib.Path(output_file_path).write_text(updated_md_text, encoding="utf-8")
    print('saved teh file in the folder output-files in the instance')
    md_text = Path(output_file_path).read_text()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    pre_upsert_df = pd.DataFrame(columns = ["id", "values", "metadata"])
    print(f'initialized an empty dataframe pre_upsert_df')
    my_splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap = 100, length_function = len)
    chunks = my_splitter.split_text(md_text)
    client=OpenAI()
    index_df = prepare_DF(my_index_df)
    print(f'did chunking adn created index_df')
  # split the data in sections


    # Call the function to load chunks into the DataFrame
    load_chunks(df=pre_upsert_df, split_text=chunks)
    pre_upsert_df.to_csv('./output-files/aws-text.csv', index=False )
    pc = Pinecone(api_key= os.environ.get('PINECONE_API_KEY'))
    print(f'created instance for Pincone and going to upload to pinecone account in vector database called test-vector-db')
    index = pc.Index('test-vector-db')
    data_path = Path('./output-files/aws-text.csv')
    my_index_df = pd.read_csv(data_path)
    print(f'saved the final file as csv in folder output-files')

    for chunk in chunker(index_df,200):
        index.upsert(vectors=convert_data(chunk))
    print(f'upsert complete , please check the website for updated vector db values')
to_be_executed('aws-overview')

'''
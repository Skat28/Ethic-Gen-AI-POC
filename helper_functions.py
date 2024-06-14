import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymupdf4llm
import pymupdf
import pymupdf4llm
import pathlib
from langchain.text_splitter import MarkdownTextSplitter
import os
import json
from openai import OpenAI
from pinecone import Pinecone
import textwrap
import pathlib 
from flask import Flask
from flask import Flask, request,redirect, url_for, send_from_directory, render_template
import re
from pathlib import Path
import ast
import time
import re
import boto3


table_pattern = re.compile(r'(\|.*?\|\n(\|.*?---.*?\|\n)(\|.*?\|\n)*)')
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

client=OpenAI()
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
    #'Converts a pandas dataframe to be a simple list of tuples, formatted how the `upsert()` method in the Pinecone Python client expects.'
    data = []
    for i in chunk.to_dict('records'):
        data.append(i)
    return data




def to_be_executed(file_name):

    table_pattern = re.compile(r'(\|.*?\|\n(\|.*?---.*?\|\n)(\|.*?\|\n)*)')
    path_to_new_file=(f'pdf-files/{file_name}.pdf')   
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

    # print('created openai client')

    print(f'did chunking adn created index_df')
  # split the data in sections


    # Call the function to load chunks into the DataFrame
    load_chunks(df=pre_upsert_df, split_text=chunks)
    pre_upsert_df.to_csv('./output-files/aws-text.csv', index=False )
    print(f'saved pre upsert df to a file')
    my_index_df=pd.read_csv('./output-files/aws-text.csv')
    index_df = prepare_DF(my_index_df)
    print(f'read teh dataframe content from csv file')
    pc = Pinecone(api_key= os.environ.get('PINECONE_API_KEY'))
    print(f'created instance for Pincone and going to upload to pinecone account in vector database called test-vector-db')
    index = pc.Index('test-vector-db')
    print(f'created index')
    data_path = Path('./output-files/aws-text.csv')
    my_index_df = pd.read_csv(data_path)
    print(f'saved the final file as csv in folder output-files')


    for chunk in chunker(index_df,200):
        index.upsert(vectors=convert_data(chunk))
    print(f'upsert complete , please check the website for updated vector db values')
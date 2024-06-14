from helper_functions import *
import boto3






def main():
    sqs = boto3.client('sqs')

    queue_url = 'https://sqs.us-east-1.amazonaws.com/239902576304/week7-gen-ai'

    # Receive message from SQS queue
    x=1
    while True:
        print(f'running while loop for {x}th time')
        x+=1
        response = sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=[
                'SentTimestamp'
            ],
            MaxNumberOfMessages=10,
            MessageAttributeNames=[
                'All'
            ],
            VisibilityTimeout=30,
            WaitTimeSeconds=10
        )
        if 'Messages' in response.keys():
            messages = response['Messages']
            processed_message_ids=[]
            i=1
            for message in messages:
                print(f'running for loop for {i}th time')
                i+=1
                current_message_id = message['MessageId']
                if current_message_id not in processed_message_ids:
                    processed_message_ids.append(current_message_id)
                    body_str = message['Body']
                    body_dict = json.loads(body_str)
                    print(f'\ncurrently processed_message_ids:{processed_message_ids}\n\n')
                    s3_bucket_info = body_dict['Records'][0]['s3']['bucket']
                    s3_content_info = body_dict['Records'][0]['s3']['object']

                    file_name=s3_content_info['key']
                    bucket_name=s3_bucket_info['name']
                    print(f'filename:{file_name} and bucket:{bucket_name}')
                    # import boto3

                    s3 = boto3.client('s3')
                    # print(f'client:{s3}')
                    try:
                        s3.download_file(Bucket=bucket_name,Key=file_name,Filename=f'./pdf-files/{file_name}')
                    except Exception:
                        continue
                    # s3.download_file(Bucket=bucket_name,Key='Acct Statement_XX3787_10062024.pdf',Filename=f'./pdf-files/Acct Statement_XX3787_10062024.pdf')
                    # breakpoint()
                    receipt_handle = message['ReceiptHandle']

                    # copy_file_from_s3_to_local(Bucket=bucket_name,CopySource ={'Bucket':bucket_name,'Key':file_name},Key=f'./pdf-files/{file_name}')

                    to_be_executed(file_name[:-4])

                    Delete received message from queue
                    sqs.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=receipt_handle
                    )
                    # time.sleep(5)
                    # breakpoint()
        else:
            print('no messages found currently')
            # print(f'message:{messages}')
        continue

if __name__ =='__main__':
    main()
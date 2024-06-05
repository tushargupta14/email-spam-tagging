import json

import boto3, datetime, os

sm = boto3.client('sagemaker')

def list_jobs(results):
    jobs = sm.list_training_jobs(MaxResults=int(results), SortBy="CreationTime",SortOrder="Descending")
    job_names = map(lambda job: [job['TrainingJobName'], job['TrainingJobStatus']],  jobs['TrainingJobSummaries'])
    return {'jobs': list(job_names)}

def get_job_by_name(name):
    job = sm.describe_training_job(TrainingJobName=name)
    return {'job': str(job)}


def create_model(name, container, model_data_url):
    """ Create SageMaker model.
    Args:
        name (string): Name to label model with
        container (string): Registry path of the Docker image that contains the model algorithm
        model_data_url (string): URL of the model artifacts created during training to download to container
    Returns:
        (None)
    """
    try:
        sm.create_model(
            ModelName=name,
            PrimaryContainer={
                'Image': container,
                'ModelDataUrl': model_data_url
            },
            ExecutionRoleArn=EXECUTION_ROLE
        )
    except Exception as e:
        print(e)
        print('Unable to create model.')
        raise(e)

def create_endpoint_config(name, model_name):
    """ Create sagemaker endpoint configuration.
    Args:
        name (string): Name to label endpoint configuration with.
    Returns:
        (None)
    """
    try:
        sm.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[
                {
                    'VariantName': 'prod',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large'
                }
            ]
        )
    except Exception as e:
        print(e)
        print('Unable to create endpoint configuration.')
        raise(e)

def create_endpoint(endpoint_name, config_name):
    """ Create sm endpoint with input endpoint configuration.
    Args:
        endpoint_name (string): Name of endpoint to create.
        config_name (string): Name of endpoint configuration to create endpoint with.
    Returns:
        (None)
    """
    try:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except Exception as e:
        print(e)
        print('Unable to create endpoint.')
        raise(e)

def lambda_handler(event, context):
    
    # Get the latest completed job
    jobs = list_jobs(2)
    for response in jobs['jobs']:
        if response[1] == "Completed":
            training_job_name = response[0]
            break
    print("Last successful training job name: ", training_job_name)
    
    job = sm.describe_training_job(TrainingJobName=training_job_name)
        
    training_job_prefix = "lambda-retrain"
    training_job_name = training_job_prefix+str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]

    print("Starting training job %s" % training_job_name)

    
    if 'VpcConfig' in job:
        resp = sm.create_training_job(
            TrainingJobName=training_job_name, 
            AlgorithmSpecification=job['AlgorithmSpecification'], 
            RoleArn=job['RoleArn'],
            InputDataConfig=job['InputDataConfig'], 
            OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], 
            StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
            VpcConfig=job['VpcConfig'],
            Tags=job['Tags'] if 'Tags' in job else [])
    else:
        # Because VpcConfig cannot be empty like HyperParameters or Tags :-/
        resp = sm.create_training_job(
            TrainingJobName=training_job_name, 
            AlgorithmSpecification=job['AlgorithmSpecification'], 
            RoleArn=job['RoleArn'],
            InputDataConfig=job['InputDataConfig'], 
            OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], 
            StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
            Tags=job['Tags'] if 'Tags' in job else [])

    print("-----Train job creation Response-----", resp)
    
    best_training_job = training_job_name
    endpoint = 'spam-prediction'
    model_data_url = 's3://sagemaker-hw3-bucket/sms-spam-classifier/output/sms-spam-classifier-mxnet-2021-04-05-18-50-21-492/output/model.tar.gz'
    model_name = 'sms-spam-classifier-mxnet-2021-04-05-18-50-21-492'
    container = '520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-mxnet:1.1-cpu-py2'
    
    print('Creating endpoint configuration...')
    create_endpoint_config(best_training_job, model_name)
    
    #create_endpoint(endpoint, best_training_job)
    
    print("Updating end point: {}".format(endpoint))
    response = sm.update_endpoint(
        EndpointName=endpoint,
        EndpointConfigName=best_training_job
    )    
    print(response)

    return None

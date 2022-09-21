import boto3, datetime, subprocess, json
from time import sleep

now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d%H%M%S')
image = 'sklearn'
tag = ':latest'
repository_name = f'{image}-{now}'


###
USE_PREEXIST_ROLE = True

# ECR のリポジトリを作成する
ecr = boto3.client('ecr')
response = ecr.create_repository(
    repositoryName=repository_name,
    imageScanningConfiguration={'scanOnPush': True},
)

# 必要な情報を抜き取っておく
uri = response['repository']['repositoryUri']
account_id = response['repository']['registryId']
region = uri.split('.')[3]
domain = uri.split('/')[0]



# build からプッシュまで

# build 済なら実行不要
!docker build -t {image} .

# タグ付与
!docker tag {image}{tag} {uri}{tag}

# ECR にログイン
! aws ecr get-login-password | docker login --username AWS --password-stdin {domain}

# push
!docker push {uri}{tag}


# 必要な情報を抜き取っておく
res = ecr.describe_images(
    repositoryName = repository_name
)
image_digest = res['imageDetails'][0]['imageDigest']


iam = boto3.client('iam')
function_name = f'{image}-function-{now}'
doc = {
    'Version': '2012-10-17',
    'Statement': [
        {
            'Action': 'sts:AssumeRole',
            'Principal': {
                'Service': 'lambda.amazonaws.com'
                
            },
            'Effect': 'Allow',
            'Sid': ''
            
        }
    ]
}

if not(USE_PREEXIST_ROLE):
    # ロール作成
    role_name = f'{image}-role-{now}'
    res = iam.create_role(
        Path = '/service-role/',
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(doc),
        Description=f'exec role',
        MaxSessionDuration=3600*12
    )
    role_arn = res['Role']['Arn']

doc = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "logs:CreateLogGroup",
            "Resource": f"arn:aws:logs:{account_id}:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                f"arn:aws:logs:{region}:{account_id}:log-group:/aws/lambda/{function_name}:*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:GetObject",
                "s3:GetObjectAcl",
                "s3:AbortMultipartUpload"
            ],
            "Resource": [
                "arn:aws:s3:::taturabe-dataset",
                "arn:aws:s3:::taturabe-dataset/*"
            ]
        }
    ]
}

if not(USE_PREEXIST_ROLE):
    # ポリシー作成
    poicy_name = f'{image}-policy-{now}'
    res = iam.create_policy(
        PolicyName=poicy_name,
        PolicyDocument=json.dumps(doc),
    )
    policy_arn = res['Policy']['Arn']
    
    # 作成したポリシーをロールにアタッチ
    res = iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy_arn
    )
    
    # IAM の反映をしばし待つ
    sleep(20)





if USE_PREEXIST_ROLE:
    # put your existing role
    role_arn = 'arn:aws:iam::820974724107:role/service-role/sklearn-role-20220915145022'


# lambda function を Docker で作成したイメージから作成する
lambda_client = boto3.client('lambda')
res = lambda_client.create_function(
    FunctionName=function_name,
    Role=role_arn,
    Code={
        'ImageUri': f'{uri}@{image_digest}'
    },
    Description='isolation forest for anomaly detection MLOps',
    Timeout=60*15,
    MemorySize=1024,
    Publish=True,
    PackageType='Image',
)
# 作成が完了するまで待つ
while True:
    res = lambda_client.get_function(FunctionName=function_name)
    try:
        if res['Configuration']['StateReasonCode']=='Creating':
            print('.',end='')
            sleep(1)
    except:
        if res['Configuration']['LastUpdateStatus']=='Successful':
            print('!')
            break
        else:
            print('?')
            break




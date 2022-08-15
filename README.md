# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

uploaded to s3://sagemaker-us-east-1-979233489196/dogImages/

![screenshot](https://raw.githubusercontent.com/MAHMOUDRR707/Image-Classification-using-AWS-SageMaker/master/Screenshots/s3%20Dataset.png?token=GHSAT0AAAAAABWYDGOUYOUKBIJFZV5ARDHUYX2ABMQ)

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

 **- for HPO.PY file  used for  retrieve the best best hyperparameters from all your training jobs**


I worked on different parameters like LR , batch_size and epochs and best of them : 

LR : 0.00604276152932198
Batch_Size  : 128
Ephocs : 3

Remember that your README should:

- Include a screenshot of completed training jobs

I start  working on 3 Trainig jobs . 

![training jobs](https://raw.githubusercontent.com/MAHMOUDRR707/Image-Classification-using-AWS-SageMaker/master/Screenshots/trainingjob.png?token=GHSAT0AAAAAABWYDGOUV3HVU7PTK7Q2XYQEYX2AF2Q)

- Tune at least two hyperparameters

objective_metric_name = "average test accuracy"

objective_type = "Maximize"

metric_definitions = [{"Name": "average test accuracy", "Regex": ([0-9\\.]+)"}]

- learning rate (lr) with a range of .001 - .01

- batch size with options 16,32,128

- epochs with a range of 3 to 5

![Tuning jobs](https://raw.githubusercontent.com/MAHMOUDRR707/Image-Classification-using-AWS-SageMaker/master/Screenshots/hyperparameter%20tuning.png?token=GHSAT0AAAAAABWYDGOVHDUCMHDWM2M62VNQYX2AHWQ)


![Tuning job sucess](https://raw.githubusercontent.com/MAHMOUDRR707/Image-Classification-using-AWS-SageMaker/master/Screenshots/hyperparameter%20tuning2.png?token=GHSAT0AAAAAABWYDGOV2FEGJSYZG4FJ5Y3CYX2AICA)

- Retrieve the best best hyperparameters from all your training jobs

best hyperparameter I got  :

LR : 0.00604276152932198

Batch_Size  : 128

Ephocs : 3

## Debugging and Profiling

 **- for train_model.py  used for makeing my debugger and profiler work. then  ran it with a new estimator and printed the results.**
 
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
I created a functioning model first with fine-tuned hyperparameters. Then I imported the configurations and rules required to set up the profiler and debugger. In order to test overfit and GPU utilisation, for example, I set the criteria and configurations accordingly. After that, I modified train_model.py as necessary to enable my debugger and profiler. The findings were printed after I eventually ran it with a new estimator.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

The results of the debugging/profiling session arose the following output:

```
2022-08-15 04:55:20 Uploading - Uploading generated training model
2022-08-15 04:55:20 Completed - Training job completed
Training seconds: 1587
Billable seconds: 1587
```

Check the debugger report   :



[Debugger Report](https://github.com/MAHMOUDRR707/Image-Classification-using-AWS-SageMaker/blob/master/profiler-debugging-report.html)

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

To deploy the model, it is required to create a python file called inference.py which loads the model and transforms the input.

 **-  for inference.py used for taking the content_type of "image/jpeg" as Tensor binary input and return the classification result, the other content_types are handled with an exception**
 
To call the model, we have to execute the following code :

```python
from PIL import Image
import io
buf = io.BytesIO()
Image.open("dogImages/test/001.Affenpinscher/Affenpinscher_00036.jpg").save(buf, format="JPEG")

response = predictor.predict(buf.getvalue())
```

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

![Endpoint screenshot](https://raw.githubusercontent.com/MAHMOUDRR707/Image-Classification-using-AWS-SageMaker/master/Screenshots/endpoint.png?token=GHSAT0AAAAAABWYDGOUVOBAVJM42HNSA7NGYX2A2DA)

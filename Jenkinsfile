pipeline {
    agent none

    environment {
        GITHUB_PAT = credentials('github-pat') // GitHub PAT for pushing files
    }

    stages {

        // STAGE 1: Data Collection
        stage('Data Collection') {
            agent {
                docker {
                    image 'python:3.11'
                    args '-u root:root'
                }
            }
            steps {
                sh '''
                pip install --upgrade pip
                pip install pandas requests dvc[s3] boto3
                python 1_data_collection.py
                dvc add data/raw_data.csv
                git config --global user.name "jenkins-bot"
                git config --global user.email "jenkins@example.com"
                git add data/raw_data.csv.dvc
                git commit -m "Update raw data [CI]"
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main
                '''
            }
        }

        // STAGE 2: Feature Engineering
        stage('Feature Engineering') {
            agent {
                docker {
                    image 'python:3.11'
                    args '-u root:root'
                }
            }
            steps {
                sh '''
                pip install pandas numpy dvc[s3]
                dvc pull data/raw_data.csv.dvc
                python 2_feature_engineering.py
                dvc add data/feature_data.csv
                git add data/feature_data.csv.dvc
                git commit -m "Update feature data [CI]"
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main
                '''
            }
        }

        // STAGE 3: Feature Selection
        stage('Feature Selection') {
            agent {
                docker {
                    image 'shaikhhumair/aqi-pipeline:latest'
                    args '-u root:root'
                }
            }
            steps {
                sh '''
                pip install dvc[s3]
                dvc pull data/feature_data.csv.dvc
                python 3_feature_selection.py
                dvc add feature_selection.csv
                git add feature_selection.csv.dvc
                git commit -m "Update feature selection output [CI]"
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main
                '''
            }
        }

        // STAGE 4: Model Save
        stage('Model Save') {
            agent {
                docker {
                    image 'shaikhhumair/aqi-pipeline:latest'
                    args '-u root:root'
                }
            }
            steps {
                sh '''
                pip install dvc[s3] boto3
                dvc pull feature_selection.csv.dvc
                python 4_model_save.py
                aws s3 cp sarimax_model_*.pkl s3://s3-bucket-umairrr/models/
                '''
            }
        }

        // STAGE 5: BentoML Forecast & Push Output
        stage('BentoML Forecast') {
            agent {
                docker {
                    image 'shaikhhumair/aqi-pipeline:latest'
                    args '-u root:root'
                }
            }
            steps {
                sh '''
                pip install bentoml==1.2.0 dvc[s3] pandas numpy scikit-learn statsmodels requests
                dvc pull feature_selection.csv.dvc
                
                MODEL=$(aws s3 ls s3://s3-bucket-umairrr/models/ | grep sarimax_model_ | sort | tail -n1 | awk '{print $4}')
                aws s3 cp "s3://s3-bucket-umairrr/models/$MODEL" ./
                
                bentoml models import "$MODEL"
                MODEL_TAG=$(bentoml models list --output=json | jq -r '.[] | select(.tag | startswith("sarimax_model:")) | .tag' | sort | tail -n1)

                cp Model\ Serving/service.py ./service.py
                nohup bentoml serve service.py:svc --port 3000 --host 0.0.0.0 > bentoml.log 2>&1 &
                
                sleep 15
                curl -X POST http://localhost:3000/health_check -H "Content-Type: application/json" -d '{}'
                python Prediction\ Client/run_prediction_client.py
                
                pkill -f "bentoml serve" || true

                git config --global user.name "jenkins-bot"
                git config --global user.email "jenkins@example.com"
                git add bentoml_forecast_output.csv
                git commit -m "Update forecast output [CI]"
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main
                '''
            }
        }
    }
}
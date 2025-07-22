pipeline {
    agent none

    environment {
        GITHUB_PAT = credentials('github-token') // GitHub PAT for pushing files
    }

    stages {

        stage('Setup Environments') {
            agent any
            environment {
                VENV_LIGHT = "${HOME}/.venv-aqi-light"
                VENV_HEAVY = "${HOME}/.venv-aqi-heavy"
            }
            steps {
                sh '''
                echo "[INFO] Setting up virtual environments..."

                # LIGHT env: for Stage 1 & 2
                if [ ! -d "$VENV_LIGHT" ]; then
                    python3 -m venv $VENV_LIGHT
                    . $VENV_LIGHT/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                else
                    echo "[INFO] Light env already exists."
                fi

                # HEAVY env: for Stage 3-5
                if [ ! -d "$VENV_HEAVY" ]; then
                    python3 -m venv $VENV_HEAVY
                    . $VENV_HEAVY/bin/activate
                    pip install --upgrade pip
                    pip install dvc[s3] boto3 s3fs pandas numpy scikit-learn statsmodels bentoml
                else
                    echo "[INFO] Heavy env already exists."
                fi
                '''
            }
        }


        // STAGE 1: Data Collection
        stage('Data Collection') {
           agent any
            environment {
                VENV_LIGHT = "${HOME}/.venv-aqi-light"
            }
            steps {
                sh '''
                echo "[INFO] Activating light virtual environment..."
                . $VENV_LIGHT/bin/activate

                # Run daily fetch script
                python fetch_daily_data.py

                # Git config and commit logic
                git config --global user.name "jenkins-bot"
                git config --global user.email "jenkins@example.com"
                git add karachi_weather_apr1_to_current.csv
                git commit -m "Daily update: AQI + Weather data" || echo "No changes to commit"
                git pull --rebase origin main || true
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main || true
            '''
            }
        }

        // STAGE 2: Feature Engineering
        stage('Feature Engineering') {
            agent any
            environment {
                VENV_LIGHT = "${HOME}/.venv-aqi-light"
            }
            steps {
                sh '''
                echo "[INFO] Activating light virtual environment..."
                . $VENV_LIGHT/bin/activate

                # Run data quality check and feature engineering scripts
                python "Data Preprocessing/data_quality_check.py"
                python "Data Preprocessing/run_preprocessing.py"

                # Git commit and push
                git config --global user.name "jenkins-bot"
                git config --global user.email "jenkins@example.com"
                git add full_preprocessed_aqi_weather_data_with_all_features.csv
                git commit -m "Daily update: Feature engineered AQI + weather data" || echo "No changes to commit"
                git pull --rebase origin main || true
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main || true
            '''
            }
        }

        // STAGE 3: Feature Selection
        stage('Feature Selection') {
            agent any
            environment {
                VENV_HEAVY = "${HOME}/.venv-aqi-heavy"
                AWS_ACCESS_KEY_ID     = credentials('aws-credentials')    // Jenkins Credentials (ID = aws-credentials)
                AWS_SECRET_ACCESS_KEY = credentials('aws-credentials')    // Same ID used for both
                AWS_DEFAULT_REGION    = 'us-east-1'
                GITHUB_PAT            = credentials('github-token')       // Jenkins Credentials (ID = github-token)
            }

            steps {
                sh '''
                echo "[INFO] Activating Heavy virtual environment..."
                . $VENV_HEAVY/bin/activate

                # Optional: re-authenticate git (in case Jenkins doesnt inherit global config)
                git config --global user.name "jenkins-bot"
                git config --global user.email "jenkins@example.com"

                # Pull latest feature_data.csv from DVC remote
                dvc pull data/feature_data.csv.dvc

                # Run feature selection
                python 3_feature_selection.py

                # Track the new/updated output
                dvc add feature_selection.csv

                # Commit & push DVC metadata
                git add feature_selection.csv.dvc .gitignore
                git commit -m "Update feature selection output [CI]" || echo "No changes to commit"
                git pull --rebase origin main || true
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main || true

                # Push data to S3
                dvc push
            '''
            }
        }

        // STAGE 4: Model Training and Serving
        stage('Model Training and Serving') {
            agent any
            environment {
                VENV_HEAVY = "${HOME}/.venv-aqi-heavy"
                AWS_ACCESS_KEY_ID     = credentials('aws-credentials')    // Jenkins Credentials (ID = aws-credentials)
                AWS_SECRET_ACCESS_KEY = credentials('aws-credentials')    // Same ID used for both
                AWS_DEFAULT_REGION    = 'us-east-1'
                GITHUB_PAT            = credentials('github-token')       // Jenkins Credentials (ID = github-token)
            }
            steps {
                sh '''
                echo "[INFO] Activating Heavy virtual environment..."
                . $VENV_HEAVY/bin/activate
                # Git config
                git config --global user.name "jenkins-bot"
                git config --global user.email "jenkins@example.com"

                # Pull feature_selection.csv from S3 (via DVC)
                dvc pull

                # Run AQI prediction script (generates predictions.csv)
                python "Model Training/predict_next_3days.py"

                # Track predictions.csv with DVC
                dvc add predictions.csv

                # Commit DVC metadata
                git add predictions.csv.dvc .gitignore
                git commit -m "Update predictions.csv via DVC [CI]" || echo "No changes to commit"
                git pull --rebase origin main || true
                git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main || true

                # Push predictions.csv to S3 via DVC
                dvc push

                # Get latest SARIMAX BentoML model tag
                MODEL_TAG=$(bentoml models list --output=json | jq -r '.[] | select(.tag | startswith("sarimax_model:")) | .tag' | sort | tail -n1)
                echo "MODEL_TAG=${MODEL_TAG}"
                
                # Export model to .bentomodel archive
                MODEL_HASH=${MODEL_TAG#*:}
                EXPORT_BASE="sarimax_model_${MODEL_HASH}"
                ACTUAL_FILENAME="${EXPORT_BASE}.bentomodel"
                bentoml models export "$MODEL_TAG" "$EXPORT_BASE"

                # Upload SARIMAX model to S3
                aws s3 cp "$ACTUAL_FILENAME" "s3://s3-bucket-umairrr/models/"

                # Clean up older SARIMAX models (keep last 3)
                files=$(aws s3 ls s3://s3-bucket-umairrr/models/ | grep sarimax_model_ | sort)
                total_files=$(echo "$files" | wc -l)

                if [ "$total_files" -le 3 ]; then
                    echo "Nothing to delete. Less than or equal to 3 models."
                else
                    to_delete=$(echo "$files" | head -n $(($total_files - 3)) | awk '{print $4}')
                    for file in $to_delete; do
                        echo "Deleting $file..."
                        aws s3 rm "s3://s3-bucket-umairrr/models/$file"
                    done
                    echo "Cleanup complete."
                fi
                '''
            }
        }

        // STAGE 5: BentoML Forecast & Push Output
        stage('BentoML Forecast') {
            agent any
            environment {
                VENV_HEAVY = "${HOME}/.venv-aqi-heavy"
                AWS_ACCESS_KEY_ID     = credentials('aws-credentials')    // Jenkins Credentials (ID = aws-credentials)
                AWS_SECRET_ACCESS_KEY = credentials('aws-credentials')    // Same ID used for both
                AWS_DEFAULT_REGION    = 'us-east-1'
                GITHUB_PAT            = credentials('github-token')       // Jenkins Credentials (ID = github-token)
            }
            
             steps {
                sh '''
                echo "[INFO] Activating Heavy virtual environment..."
                . $VENV_HEAVY/bin/activate
                
                echo "Pulling feature_selection.csv from S3 via DVC..."
                dvc pull feature_selection.csv.dvc
                
                echo "Getting latest SARIMAX model from S3..."
                LATEST_MODEL=$(aws s3 ls s3://s3-bucket-umairrr/models/ | grep sarimax_model_ | sort | tail -n1 | awk '{print $4}')
                echo "Latest model: $LATEST_MODEL"
                
                if [ -z "$LATEST_MODEL" ]; then
                    echo "No SARIMAX model found in S3"
                    exit 1
                fi
                
                echo "Downloading $LATEST_MODEL..."
                aws s3 cp "s3://s3-bucket-umairrr/models/$LATEST_MODEL" ./
                
                echo "Importing BentoML model: $LATEST_MODEL"
                bentoml models import "$LATEST_MODEL"
                
                MODEL_TAG=$(bentoml models list --output=json | jq -r '.[] | select(.tag | startswith("sarimax_model:")) | .tag' | sort | tail -n1)
                echo "Imported model tag: $MODEL_TAG"
                
                echo "Copying service file..."
                cp "Model Serving/service.py" ./service.py
                echo "Copied service.py from Model Serving folder"
                
                echo "Starting BentoML service with model: $MODEL_TAG"
                nohup bentoml serve service.py:svc --port 3000 --host 0.0.0.0 > bentoml_service.log 2>&1 &
                
                echo "Waiting for BentoML service to start..."
                for i in {1..30}; do
                    if curl -s http://localhost:3000/health_check -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
                        echo "BentoML service is running"
                        break
                    fi
                    echo "Attempt $i: Service not ready yet, waiting..."
                    sleep 3
                done
                
                # Check if service is actually running
                if ! curl -s http://localhost:3000/health_check -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
                    echo "BentoML service failed to start"
                    echo "Service logs:"
                    cat bentoml_service.log || echo "No log file found"
                    echo "Process list:"
                    ps aux | grep bentoml || true
                    exit 1
                fi
                
                echo "Testing service..."
                curl -X POST http://localhost:3000/health_check -H "Content-Type: application/json" -d '{}'
                
                echo "Running BentoML prediction client..."
                python "Prediction Client/run_prediction_client.py"
                '''
            }
            post {
                always {
                    sh '''
                    echo "Stopping BentoML service..."
                    pkill -f "bentoml serve" || true
                    echo "Service stopped"
                    '''
                }
                success {
                    sh '''
                    echo "Committing updated forecast output to GitHub..."
                    git config --global user.name "jenkins-bot"
                    git config --global user.email "jenkins@example.com"
                    git add bentoml_forecast_output.csv
                    git commit -m "Update bentoml_forecast_output.csv [CI]" || echo "No changes to commit"
                    git push https://${GITHUB_PAT}@github.com/uma1r111/10pearls-AQI-Project-.git HEAD:main
                    '''
                }
            }
        }
    }
}
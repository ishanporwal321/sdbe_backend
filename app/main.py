import datetime
import logging
import openai
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from passlib.context import CryptContext
import boto3
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from uuid import uuid4
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from dotenv import load_dotenv
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # This loads the variables from .env

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure AWS S3 Client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = "ishansdsignupbucket"  # Replace with your S3 bucket name

# Load the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS Index for storing job embeddings
dimension = 384  # Dimension of the embeddings from the model
index = None


# Load job data from a JSON file
def load_jobs(file_name="jobs.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    with open(file_path, "r") as f:
        jobs = json.load(f)
    logging.info(f"Loaded {len(jobs)} jobs from {file_path}")
    return jobs

# Convert job descriptions into embeddings and store them in FAISS
def store_job_embeddings_in_faiss(jobs):
    global index
    job_embeddings = []
    for job in jobs:
        combined_description = f"{job['title']} at {job['company']}, located in {job['location']}. Description: {job['description']}"
        embedding = model.encode(combined_description)
        job_embeddings.append(embedding)
    job_embeddings = np.array(job_embeddings)
    index = faiss.IndexFlatL2(dimension)
    index.add(job_embeddings)
    logging.info(f"Added {len(job_embeddings)} embeddings to FAISS index")

def initialize_jobs_and_index():
    global jobs_data, index
    if os.path.exists('jobs_data.pkl') and os.path.exists('faiss_index.pkl'):
        with open('jobs_data.pkl', 'rb') as f:
            jobs_data = pickle.load(f)
        with open('faiss_index.pkl', 'rb') as f:
            index = pickle.load(f)
        logging.info("Loaded jobs data and FAISS index from pickle files")
    else:
        jobs_data = load_jobs()
        store_job_embeddings_in_faiss(jobs_data)
        with open('jobs_data.pkl', 'wb') as f:
            pickle.dump(jobs_data, f)
        with open('faiss_index.pkl', 'wb') as f:
            pickle.dump(index, f)
        logging.info("Created and saved jobs data and FAISS index to pickle files")

initialize_jobs_and_index()
    
# Pre-load job data and add to FAISS index
initialize_jobs_and_index()

# Function to transcribe audio to text
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)
            return transcription
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand the audio")
        return ""
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# Function to ask OpenAI why the candidate is a good fit for the job
def ask_openai_about_fit(transcription, job_description):
    prompt = f"Given the candidate's description:\n{transcription}\n\nAnd the job description:\n{job_description}\n\nExplain why the candidate would be a good fit for this job."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a job fit evaluator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"Error with OpenAI API: {str(e)}")
        return "Unable to process with OpenAI."


@app.post("/record-job-click")
async def record_job_click(job_url: str = Form(...), username: str = Form(...), token: str = Depends(oauth2_scheme)):
    try:
        # Get the user data
        user_key = f"users/{username}.json"
        logging.info(f"Attempting to get user data for username: {username}")
        try:
            user_response = s3.get_object(Bucket=BUCKET_NAME, Key=user_key)
            user_data = json.loads(user_response['Body'].read().decode('utf-8'))
            logging.info(f"Retrieved user data for username: {username}")
        except s3.exceptions.NoSuchKey:
            logging.error(f"User data not found for username: {username}")
            raise HTTPException(status_code=404, detail=f"User not found: {username}")
        except Exception as e:
            logging.error(f"Error retrieving user data: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving user data")

        # Verify the token matches the user_id in the user data
        if user_data['user_id'] != token:
            logging.error(f"Token mismatch for user {username}")
            raise HTTPException(status_code=401, detail="Invalid token")

        # Create or update the user's job clicks file
        clicks_key = f"user_job_clicks/{username}.json"
        logging.info(f"Attempting to get job clicks for username: {username}")
        try:
            clicks_response = s3.get_object(Bucket=BUCKET_NAME, Key=clicks_key)
            clicks_data = json.loads(clicks_response['Body'].read().decode('utf-8'))
            logging.info(f"Retrieved existing job clicks for {username}")
        except s3.exceptions.NoSuchKey:
            logging.info(f"No existing job clicks found for {username}, creating new list")
            clicks_data = []
        except Exception as e:
            logging.error(f"Error retrieving job clicks: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving job clicks")

        # Add the new job URL to the list
        clicks_data.append({
            "url": job_url,
            "timestamp": datetime.datetime.now().isoformat()
        })
        logging.info(f"Added new job click for {username}: {job_url}")

        # Store the updated clicks data back in S3
        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=clicks_key,
                Body=json.dumps(clicks_data),
                ContentType='application/json'
            )
            logging.info(f"Successfully stored updated job clicks for {username}")
        except Exception as e:
            logging.error(f"Error storing updated job clicks: {str(e)}")
            raise HTTPException(status_code=500, detail="Error storing job clicks")

        return {"message": "Job click recorded successfully"}
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in record_job_click: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")





# Route to handle user behavioral questions, audio file, and match jobs
@app.post("/submit-answers")
async def submit_answers(
    question1: str = Form(...),
    question2: str = Form(...),
    question3: str = Form(...),
    user_id: str = Form(...),
    audio: UploadFile = File(None)
):
    try:
        # Transcribe the audio file if present
        transcription = ""
        if audio:
            audio_data = BytesIO(await audio.read())
            audio_segment = AudioSegment.from_file(audio_data, format="wav")

            # Log the audio metadata for inspection
            logging.info(f"Channels: {audio_segment.channels}, Frame Rate: {audio_segment.frame_rate}, Duration: {len(audio_segment)} ms")

            # Save the received audio for debugging purposes
            audio_segment.export("received_audio.wav", format="wav")
            logging.info("Audio file saved for inspection: received_audio.wav")

            # Perform transcription
            transcription = transcribe_audio("received_audio.wav")
            logging.info(f"Transcription: {transcription}")

        # Combine the user's answers and transcribed audio
        user_answers = {
            "user_id": user_id,
            "question1": question1,
            "question2": question2,
            "question3": question3,
            "transcription": transcription
        }

        # Generate a unique identifier for this submission
        submission_id = str(uuid4())

        # Store the answers in S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"submissions/{user_id}/{submission_id}.json",
            Body=json.dumps(user_answers),
            ContentType='application/json'
        )

        # Generate matches based on the answers
        matched_jobs, openai_fits = generate_matches(user_answers)

        return {
            "message": "Answers submitted successfully",
            "submission_id": submission_id,
            "matched_jobs": matched_jobs,
            "fit_explanations": openai_fits,
        }

    except Exception as e:
        logging.error(f"Error processing answers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing answers: {str(e)}")

def generate_matches(user_answers):
    # Combine all answers into a single string for embedding
    global index, jobs_data
    combined_answers = f"{user_answers['question1']} {user_answers['question2']} {user_answers['question3']} {user_answers['transcription']}"
    
    user_embedding = model.encode(combined_answers)
    user_embedding = np.array([user_embedding])

    # Search for the top 5 job matches using FAISS
    D, I = index.search(user_embedding, k=5)

    # Retrieve job details based on the indices
    matched_jobs = [jobs_data[i] for i in I[0]]

    # Generate fit explanations
    openai_fits = []
    for job in matched_jobs:
        job_description = f"{job['title']} at {job['company']}, located in {job['location']}. Description: {job['description']}"
        fit_explanation = ask_openai_about_fit(user_answers['transcription'], job_description)
        openai_fits.append({
            "job_title": job["title"],
            "fit_explanation": fit_explanation
        })

    return matched_jobs, openai_fits

# Root route
@app.get("/")
async def read_root():
    return {"message": "Hello, this is the root path!"}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User model for signup
class UserSignup(BaseModel):
    username: str
    email: str
    password: str

# User model for database
class UserInDB(UserSignup):
    user_id: str
    hashed_password: str

# Function to hash password
def get_password_hash(password):
    return pwd_context.hash(password)

# Signup endpoint
@app.post("/signup")
async def signup(user: UserSignup):
    # Check if user already exists (by username)
    user_key = f"users/{user.username}.json"
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=user_key)
        raise HTTPException(status_code=400, detail="Username already exists")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise

    # Create new user
    user_id = str(uuid4())
    hashed_password = get_password_hash(user.password)
    
    user_data = UserInDB(
        user_id=user_id,
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        password=user.password  # This will be excluded when converting to JSON
    )

    # Store user data in S3
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=user_key,
        Body=user_data.json(exclude={"password"}),
        ContentType='application/json'
    )

    return {"message": "User created successfully", "user_id": user_id}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

@app.post("/signin")
async def signin(form_data: OAuth2PasswordRequestForm = Depends()):
    user_key = f"users/{form_data.username}.json"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=user_key)
        user_data = json.loads(response['Body'].read().decode('utf-8'))
        if verify_password(form_data.password, user_data['hashed_password']):
            logging.info(f"User {form_data.username} signed in successfully")
            return {
                "access_token": user_data['user_id'],
                "token_type": "bearer",
                "username": form_data.username  # Add this line
            }
        else:
            logging.warning(f"Incorrect password for user {form_data.username}")
            raise HTTPException(status_code=400, detail="Incorrect username or password")
    except s3.exceptions.NoSuchKey:
        logging.warning(f"User not found: {form_data.username}")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    except Exception as e:
        logging.error(f"Error during signin for user {form_data.username}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during sign in")

@app.get("/user-answers")
async def get_user_answers(token: str = Depends(oauth2_scheme)):
    user_id = token  # In this simple example, the token is the user_id
    try:
        # List objects in the user's submissions folder
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"submissions/{user_id}/")
        
        if 'Contents' not in response:
            return {"message": "No submissions found for this user"}
        
        # Get the most recent submission
        latest_submission = max(response['Contents'], key=lambda x: x['LastModified'])
        
        # Retrieve the submission data
        submission_response = s3.get_object(Bucket=BUCKET_NAME, Key=latest_submission['Key'])
        submission_data = json.loads(submission_response['Body'].read().decode('utf-8'))
        
        return submission_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user answers: {str(e)}")

class AnswerUpdate(BaseModel):
    question1: str
    question2: str
    question3: str

@app.post("/update-answers")
async def update_answers(answers: AnswerUpdate, token: str = Depends(oauth2_scheme)):
    user_id = token  # In this simple example, the token is the user_id
    try:
        # List objects in the user's submissions folder
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"submissions/{user_id}/")
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail="No submissions found for this user")
        
        # Get the most recent submission
        latest_submission = max(response['Contents'], key=lambda x: x['LastModified'])
        
        # Retrieve the submission data
        submission_response = s3.get_object(Bucket=BUCKET_NAME, Key=latest_submission['Key'])
        submission_data = json.loads(submission_response['Body'].read().decode('utf-8'))
        
        # Update the answers
        submission_data.update(answers.dict())
        
        # Store the updated answers back in S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=latest_submission['Key'],
            Body=json.dumps(submission_data),
            ContentType='application/json'
        )
        
        return {"message": "Answers updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating answers: {str(e)}")

# Add this new endpoint for updating audio
@app.post("/update-audio")
async def update_audio(audio: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    user_id = token  # In this simple example, the token is the user_id
    try:
        # List objects in the user's submissions folder
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"submissions/{user_id}/")
        
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail="No submissions found for this user")
        
        # Get the most recent submission
        latest_submission = max(response['Contents'], key=lambda x: x['LastModified'])
        
        # Retrieve the submission data
        submission_response = s3.get_object(Bucket=BUCKET_NAME, Key=latest_submission['Key'])
        submission_data = json.loads(submission_response['Body'].read().decode('utf-8'))
        
        # Transcribe the new audio file
        audio_data = BytesIO(await audio.read())
        audio_segment = AudioSegment.from_file(audio_data, format="wav")
        audio_segment.export("temp_audio.wav", format="wav")
        new_transcription = transcribe_audio("temp_audio.wav")
        os.remove("temp_audio.wav")  # Clean up temporary file
        
        # Update the transcription
        submission_data['transcription'] = new_transcription
        
        # Store the updated data back in S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=latest_submission['Key'],
            Body=json.dumps(submission_data),
            ContentType='application/json'
        )
        
        return {"message": "Audio updated successfully", "transcription": new_transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating audio: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    # Perform any necessary cleanup here
    pass

@app.post("/generate-matches")
async def generate_matches_endpoint(user_answers: dict, token: str = Depends(oauth2_scheme)):
    user_id = token  # In this simple example, the token is the user_id
    try:
        # Generate matches based on the answers
        matched_jobs, openai_fits = generate_matches(user_answers)

        return {
            "matched_jobs": matched_jobs,
            "fit_explanations": openai_fits,
        }
    except Exception as e:
        logging.error(f"Error generating matches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating matches: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

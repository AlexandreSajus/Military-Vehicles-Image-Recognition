FROM python:3.11 as taipy

# Go to the dedicated folder and add the python corresponding folder in PATH
WORKDIR /home/taipy
ENV PATH="${PATH}:/home/taipy/.local/bin"

# Copy everything (except what is filtered by .dockerignore)
COPY . .

# Install project dependencies
RUN python3 -m pip install -r requirements.txt

# Start up command
ENTRYPOINT [ "python3", "main.py", "-H", "0.0.0.0", "-P", "5000" ]

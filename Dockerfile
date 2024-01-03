# Specify base image
FROM python:3.7-slim-stretch

# Install dependencies
#RUN apt-get update
#RUN apt-get install -y --no-install-recommends
#RUN build-essential make gcc gnupg
#RUN python3-dev unixodbc-dev

# Create /app dir and it set as working directory
RUN mkdir /app
WORKDIR /app

# Copy file
COPY requirements.txt /app

# Install requirements
RUN pip install jupyter
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy relevant files and directories
COPY . /app


# Expose port and run web server
EXPOSE 8856

CMD ["jupyter", "notebook", "--port=8856", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
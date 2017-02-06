FROM node:boron

# Install dependencies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q hunspell

# Create app directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Install node dependencies
COPY package.json /usr/src/app/
RUN npm install

# Bundle app source
COPY . /usr/src/app

EXPOSE 3002
CMD [ "npm", "start" ]
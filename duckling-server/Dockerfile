FROM clojure
EXPOSE 3001

# Move code into app directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

# start duckling server
CMD lein run src/duckling_server/core.clj
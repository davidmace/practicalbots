(ns duckling-server.core
  (:require [duckling.core :as p] [clojure.data.json :as json])
  (:use org.httpkit.server)
  (:gen-class))

;; Hacky way to decode URL parameters because I didn't want to load a whole library
(defn decode-params [s]
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace
(clojure.string/replace s
#"%20" " ")
#"\+" " ")
#"%23" "#")
#"%24" "\\$")
#"%40" "@")
#"%3F" "?")
#"%3A" ":")
#"%3B" ";")
#"%2F" "/")
#"%25" "%"))

;; Ensure query string has 's' parameter then parse it through Duckling
;; For different language support, change :en$core to the values in Duckling's documentation
(defn parse_query_string [s]
  (if (and (not (nil? s)) (= (subs s 0 1) "s"))
    (json/write-str (p/parse :en$core (java.net.URLDecoder/decode (subs s 2)) [:time]))
    ""))

;; Called on all server endpoint requests
(defn app [req]
  {:status  200
   :headers {"Content-Type" "text/html"}
   :body   (parse_query_string (:query-string req))})

;; Load duckling an start server
(defn -main
  "Load duckling and start server"
  [& args]
  (p/load!) ;; Load all languages
  (run-server app {:port 3001}))


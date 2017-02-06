(defproject duckling-server "0.1.0-SNAPSHOT"
  :description "Simple server to wrap Duckling"
  :dependencies [[org.clojure/clojure "1.8.0"] [wit/duckling "0.4.0"] [http-kit "2.1.18"] 
  	[compojure "1.5.0"] [org.clojure/data.json "0.2.6"] [javax.servlet/servlet-api "2.5"]]
  :plugins [[lein-exec "0.3.6"]]
  :main ^:skip-aot duckling-server.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
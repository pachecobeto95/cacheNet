from appSetEdge import app
import config

app.debug = config.DEBUG
app.run(host=config.HOST_SET_EDGE, port=config.PORT_SET_EDGE)
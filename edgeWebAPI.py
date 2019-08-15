"""
Author: Roberto Pacheco
Date: 06/08/2019
Last Modified: 06/08/2019
Objective: send a data to the edge node
"""


from appEdge import app
import config

app.debug = config.DEBUG
app.run(host=config.HOST_EDGE, port=config.PORT_EDGE)
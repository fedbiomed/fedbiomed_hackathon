import unittest

import random
import sys
import threading
import time

import testsupport.mock_researcher_environ

from fedbiomed.researcher.environ import MQTT_BROKER, MQTT_BROKER_PORT, RESEARCHER_ID
from fedbiomed.common.messaging   import Messaging, MessagingType
from fedbiomed.common.message     import ResearcherMessages

class TestMessagingResearcher(unittest.TestCase):
    '''
    Test the Messaging class
    '''


    # once in test lifetime
    @classmethod
    def setUpClass(cls):

        random.seed()
        # verify that a broker is available
        try:
            print("connecting to:", MQTT_BROKER, "/", MQTT_BROKER_PORT)
            cls._m = Messaging(cls.on_message,
                               MessagingType.RESEARCHER,
                               RESEARCHER_ID,
                               MQTT_BROKER,
                               MQTT_BROKER_PORT)

            cls._m.start()
            cls._broker_ok = True
        except Exception as e:
            cls._broker_ok = False

        time.sleep(1.0) # give some time to the thread to listen to the given port

        pass


    @classmethod
    def tearDownClass(cls):
        if not cls._broker_ok:
            cls._m.stop()
        pass


    # before all the tests
    def setUp(self):
        pass


    # after all the tests
    def tearDown(self):
        pass


    # mqqt callbacks
    def on_message(msg, topic):
        print("RECV:", topic, msg)


    # tests
    def test_messaging_00_send(self):
        '''
        send a message on MQTT
        '''
        print("connexion status =" , self._broker_ok)
        if not self._broker_ok:
            self.skipTest('no broker available for this test')

        try:
            ping = ResearcherMessages.request_create(
                {'researcher_id' : RESEARCHER_ID,
                 'sequence'      : random.randint(1, 65535),
                 'command'       :'ping'}).get_dict()
            self._m.send_message(ping)
            self.assertTrue( True, "ping message correectly sent")

        except:
            self.assertTrue( False, "ping message not sent")


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
import os
import unittest
from ektelo import util

LOG_PATH = os.environ['EKTELO_LOG_PATH']
LOG_LEVEL = os.environ['EKTELO_LOG_LEVEL']


class TestCommon(unittest.TestCase):

    def setUp(self):
        util.setup_logging('ektelo', os.path.join(LOG_PATH, 'test.out'), LOG_LEVEL)

        self.maxDiff = None

    def tearDown(self):
        pass

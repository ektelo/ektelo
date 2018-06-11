import numpy as np
from ektelo.client.service import ProtectedDataSource
from ektelo.data import DataManager
from ektelo.private.kernel import PrivateManager
from ektelo.private.service import KernelService
from ektelo.private.transformation import Null, Vectorize
import unittest


class TestPrivate(unittest.TestCase):

    def setUp(self):
        self.n = 8
        self.data_manager = DataManager(Null())
        source_uri = 'file:///path_to_data.csv'
        private_manager = PrivateManager(source_uri, None)
        self.kernel_service = KernelService(private_manager)
        self.kernel_service.private_manager._load_data = lambda source_uri: np.ones((n,))

    def test_protected_data_source(self):
        x = ProtectedDataSource(self.kernel_service, self.data_manager)
        x2 = x.transform(Vectorize.__name__, {'name': ''})

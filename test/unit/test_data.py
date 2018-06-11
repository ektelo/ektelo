from ektelo.data import Graph
from ektelo.data import Node
import networkx as nx
import unittest


class TestData(unittest.TestCase):

    def setUp(self):
        pass

    def test_graph(self):
        n1 = Node()
        n2 = Node()
        n3 = Node()

        graph = Graph()
        graph.insert(n1)
        graph.insert(n2)
        graph.insert(n3, after=n1)

        self.assertEqual(sorted(graph.graph.edges()), 
                         sorted([(n1.id, n2.id), (n1.id, n3.id)]))

from ektelo import util
from ektelo.mixins import Marshallable
import hashlib


class Base(Marshallable):

    def __init__(self, short_name=None):
        if short_name is None:
            self.short_name = 'ektelo_' + self.__class__.__name__
        else:
            self.short_name = short_name

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.short_name))

        for key, value in util.prepare_for_hash(self.init_params).items():
            m.update(key.encode('utf-8'))
            m.update(repr(value).encode('utf-8'))

        return m.hexdigest()

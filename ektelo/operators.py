class Operator:

    def __init__(self):
        raise NotImplementedError


class SelectionOperator(Operator):

    def __init__(self):
        pass

    def select(self):
        pass


class InferenceOperator(Operator):

    def __init__(self):
        pass

    def infer(self):
        pass


class MapperOperator(Operator):

    def __init__(self):
        pass


class MeasurementOperator(Operator):
    pass


class MetaOperator(Operator):
    pass


class TransformationOperator(Operator):
    pass

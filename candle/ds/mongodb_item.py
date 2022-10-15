class Item:
    def to_dict(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        raise NotImplementedError()

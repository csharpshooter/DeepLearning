class Base:
    def __init__(self, module, device, classes):
        self.module, self.device = module, device
        self.handles = []
        self.classes = classes

    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, inputs, layer, *args, **kwargs):
        return inputs, {}
from typing import Union, List

FAKE_CONTEXT_VALUE = "MY_CONTEXT"

class FakeSecaggContext:
    def __init__(self, parties: List[str]):
        self.parties = parties
        self.stat = False
        self.cont = None
        self.success = True

    def status(self) -> bool:
        return self.stat

    def context(self) -> Union[dict, None]:
        return self.cont

    def setup(self, timeout: float = 0) -> bool:
        self.stat = self.success
        if self.stat:
            self.cont = FAKE_CONTEXT_VALUE
        else:
            self.cont = None
        return self.stat

    def set_setup_success(self, success: bool = True):
        """Choose whether next `setup`s will fail or succeed.

        This does not exist in real `SecaggContext` and is added for mocking purpose.
        """
        self.success = success


class FakeSecaggServkeyContext(FakeSecaggContext):
    pass


class FakeSecaggBiprimeContext(FakeSecaggContext):
    pass
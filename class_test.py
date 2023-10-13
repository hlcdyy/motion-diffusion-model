
class A:
    def __init__(self, x) -> None:
        self.x = x

    def printx(self):
        print(self.x)

    def printx2(self):
        print(self.x+2)

class B(A):
    def __init__(self, x) -> None:
        super().__init__(x)
    
    def printx(self):
        print(self.x+1)
    
    def wrapx(self):
        self.x += 1

class C(B):
    def __init__(self, x) -> None:
        super().__init__(x)
    
    def printx(self):
        self.wrapx()
        self.printx2()

a = A(1)
b = B(1)
c = C(1)

a.printx()
b.printx()
c.printx()

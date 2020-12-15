def sum(a, b):
  return a + b

def f(**args):
  return sum(**args)

if __name__ == "__main__":
  f(a=3,b=4)
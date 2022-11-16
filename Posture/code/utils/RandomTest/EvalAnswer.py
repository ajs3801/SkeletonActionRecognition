def EvalAnswer(name):
  strs = name.split('_')
  target = strs[1]

  strs = target.split('.')
  target = strs[0]

  return target
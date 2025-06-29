import abc
import torch
import torch.nn as nn


class Sin(nn.Module):
  """An element-wise sin activation wrapped as a nn.Module.

  Shape:
      - Input: `(N, *)` where `*` means, any number of additional dimensions
      - Output: `(N, *)`, same shape as the input

  Examples:
      >>> m = Sin()
      >>> input = torch.randn(2)
      >>> output = m(input)
  """

  def forward(self, input):
    return torch.sin(input)  # simply apply already implemented sin


class Model(nn.Module):
  """
  Constructs a fully-connected neural network with flexible depth, width and
  activation function choices.
  """

  def __init__(
      self, dimList, actType="Tanh", output_activation=nn.Identity,
      verbose=False
  ):
    """
    Initalizes the neural network with dimension of each layer and the
    following activation layer.
    """
    super(Model, self).__init__()
    self.moduleList = nn.ModuleList()
    numLayer = len(dimList) - 1
    for idx in range(numLayer):
      i_dim = dimList[idx]
      o_dim = dimList[idx + 1]

      self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
      if idx == numLayer - 1:  # final linear layer, no act.
        self.moduleList.append(output_activation())
      else:
        if actType == "Sin":
          self.moduleList.append(Sin())
        elif actType == "Tanh":
          self.moduleList.append(nn.Tanh())
        elif actType == "ReLU":
          self.moduleList.append(nn.ReLU())
        else:
          raise ValueError(
              "Activation type ({:s}) is not included!".format(actType)
          )
    if verbose:
      print(self.moduleList)

  def forward(self, x):
    for m in self.moduleList:
      x = m(x)
    return x


# == Scheduler ==
class _scheduler(abc.ABC):
  """
  The parent class for schedulers. It implements some basic functions that will
  be used in all scheduler.
  """

  def __init__(self, last_epoch=-1, verbose=False):
    """Initializes the scheduler with the index of last epoch.
    """
    self.cnt = last_epoch
    self.verbose = verbose
    self.variable = None
    self.step()

  def step(self):
    """Updates the index of the last epoch and the variable.
    """
    self.cnt += 1
    value = self.get_value()
    self.variable = value

  @abc.abstractmethod
  def get_value(self):
    raise NotImplementedError

  def get_variable(self):
    """Returns the variable.
    """
    return self.variable


class StepLR(_scheduler):
  """This scheduler will decay to end value periodically.
  """

  def __init__(
      self, initValue, period, decay=0.1, endValue=0., last_epoch=-1,
      verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes."""
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    super(StepLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable."""
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.initValue * (self.decay**numDecay)
    if self.endValue is not None and tmpValue <= self.endValue:
      return self.endValue
    return tmpValue


class StepLRMargin(_scheduler):

  def __init__(
      self, initValue, period, goalValue, decay=0.1, endValue=1, last_epoch=-1,
      verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes."""
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    self.goalValue = goalValue
    super(StepLRMargin, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable."""
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.goalValue - (self.goalValue
                                 - self.initValue) * (self.decay**numDecay)
    if self.endValue is not None and tmpValue >= self.endValue:
      return self.endValue
    return tmpValue


class StepResetLR(_scheduler):

  def __init__(
      self, initValue, period, resetPeriod, decay=0.1, endValue=0,
      last_epoch=-1, verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes."""
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    self.resetPeriod = resetPeriod
    super(StepResetLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable."""
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.initValue * (self.decay**numDecay)
    if self.endValue is not None and tmpValue <= self.endValue:
      return self.endValue
    return tmpValue

  def step(self):
    """
    Updates the index of the last epoch and the variable. It overrides the same
    function in the parent class.
    """
    self.cnt += 1
    value = self.get_value()
    self.variable = value
    if (self.cnt + 1) % self.resetPeriod == 0:
      self.cnt = -1

from dataclasses import dataclass

from hoshmap import idict

colors = ["blue", "green", "red", "gray", "yellow"]


@dataclass
class Plot:
    d: idict
    projection: str
    xlabel: str
    ylabel: str
    legend: bool
    plt: callable
    fontsize: int = 16
    marksize: int = 16

    def __post_init__(self):
        self.fig = self.plt.figure()
        self.plt.title(f"{self.projection}", fontsize=self.fontsize)
        self.ax = self.fig.gca()
        # ax.set_yscale('log')

    def __lshift__(self, other):
        slabel, color = other
        self.ax.scatter(self.d[self.xlabel], self.d[slabel], s=self.marksize, c=color, label=slabel)

    def finish(self):
        if self.legend:
            self.ax.legend(fontsize=self.fontsize)
        self.plt.xlabel(self.xlabel, fontsize=self.fontsize)
        self.plt.ylabel(self.ylabel, fontsize=self.fontsize)
        self.plt.tight_layout()
        figManager = self.plt.get_current_fig_manager()
        figManager.window.showMaximized()
        figManager.window.setWindowTitle(f"{self.projection}     {self.xlabel}")

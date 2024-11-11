class Integer:

    # The following do not exist in sage. Compare sage.rings.rational.pyi.
    #
    ####################################################################
    def __lt__(self, other) -> bool:
        ...
    ####################################################################

    def __int__(self) -> int:
        ...

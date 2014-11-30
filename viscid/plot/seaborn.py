"""Graceful way to apply seaborn to mpl if you have it installed

This is mostly controlled with parameters in your rc script

After activate_from_viscid is called, if the seaborn import is
successful, then the entire namespace is imported here and
seaborn can be used from this module.
"""

import imp

from viscid import logger

SEABORN_ACTIVATED = False
try:
    imp.find_module('seaborn')
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

##############################################
# Global config; override before importing
# mpl or calling activate_from_visicd either
#  in script, or in rc file
##############################################
enabled = False
context = "notebook"
style = "darkgrid"
palette = "deep"
font = "sans-serif"
font_scale = 1
rc = {}  # this one can't be set from rc file
##############################################

def activate_from_viscid():
    """You should not need to call this

    This function is called by viscid.plot.mpl at time of import. All
    you need to do is set the global config options above before
    importing viscid.plot.mpl. This can be done with the rc file.
    """
    if enabled:
        try:
            import seaborn as real_seaborn
            # just some fancyness so i can specify an arbitrary
            # color palette function with arbitrary arguments
            # from the rc file
            _palette_func = real_seaborn.color_palette
            ##### NOPE, letting the user run an attribute of
            ##### seaborn might be a security problem... I could
            ##### remove "__" and the like, but even still
            if isinstance(palette, (list, tuple)):
                _palette = palette
                # ##### NOPE, letting the user run an attribute of
                # ##### seaborn might be a security problem...
                # try:
                #     func_name = palette[0].strip().strip("_")
                #     func = getattr(real_seaborn, func_name)
                #     if hasattr(func, "__call__"):
                #         _palette_func = func
                #         _palette = palette[1:]
                # except (AttributeError, TypeError):
                #     pass
                # #####
            else:
                _palette = [palette]
            _palette = _palette_func(*_palette)

            # Ok, now set the defaults
            real_seaborn.set(context=context, style=style,
                             palette=_palette, font=font,
                             font_scale=font_scale, rc=rc)

            # now pull in the public namespace
            g = globals()
            for key, value in real_seaborn.__dict__.items():
                if not key.startswith("_"):
                    g[key] = value
            g['SEABORN_ACTIVATED'] = True
        except ImportError:
            logger.warn("seaborn package not installed")

##
## EOF
##

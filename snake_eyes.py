from configparser import ConfigParser, ExtendedInterpolation

def read_config(configPath, globalVars = globals(), localVars = locals()):
	'''
	Wrapper for the Python config parser to read an ini config file and return
	a dictionary of typed parameters. For documentation of Python configparser
	and ini use, see https://docs.python.org/3.8/library/configparser.html

	Expects a valid filepath to the config file as input
	'''

	params = dict()
	config = ConfigParser(inline_comment_prefixes=('#'),interpolation=ExtendedInterpolation())
	config.optionxform = lambda option: option
	config.read_file(open(configPath))
	for section in config:
		params[section] = dict()
		for key in config[section]:
			params[section][key] = interpolate(config[section],key, globalVars, localVars)
	return params

def interpolate(config, key, globalVars, localVars):
	'''
	Attempts to interpolate parameters to more useful data types based on 
	semi-intelligent methods provided by configparser. 
	
	Type precedence:
	int
	float
	boolean
	expression
	string

	Ambiguous numerical parameters default to ints. Floats are identified if
	the float and int values differ (so 1.0 would be cast to an int). 0 and 1
	are interpreted as ints instead of boolean values under the assumption that
	this doesn't impact logical operations on the values. If boolean, float,
	and	int types fail, the parameter is assumed to be a string type. An
	attempt is made to evaluate the string as a Python expression. If 
	successful, the expresison result is returned. Otherwise, the parameter is
	assumed to actually be a string.

	Floats accept scientific notation such as 1E3 for 1000

	Booleans accept a range of (case-insensitive) values: 
	True/False
	yes/no
	on/off
	1/0 (though this one is converted to int as documented above)
	'''
	try:
		floatNum = config.getfloat(key)
		try:
			intNum = config.getint(key)
			if floatNum == intNum:
				return intNum
			else:
				return floatNum
		except:
			return floatNum
	except:
		pass
	
	try:
		return config.getboolean(key)
	except:
		pass
	
	try:
		return eval(config.get(key).replace('\n',''), globalVars, localVars) # evaluate expressions
	except:
		pass
	
	return config.get(key) # returns a string


if __name__ == '__main__':
	print(read_config("./samples/example.cfg"))
#!/bin/bash

input="bad"
while [[ $input == "bad" ]]; do
	echo "Please input your auburn username: "
	read auName
	echo "Is $auName correct? (y/n)"
	read answer
	if [[ "$answer" =~ [Yy][eE]?[sS]? ]]; then 
		echo $auName >> readyToSubmit.txt
		input="good enough"
	fi
done

echo "Your code has been finalized! Your submission will be eligible for grading when you push to Master."
echo "If you ran this script on accident, simply delete readyToSubmit.txt"

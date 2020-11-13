#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

def main(input_filename, star_filename, output_filename):
	all_star = [star for star in eval(open(star_filename, 'r').readline())]
	output_file = open(output_filename, 'w')
	for line in open(input_filename, 'r').readlines():
		lineArr = line.strip().split('\001')
		if lineArr[0].strip() in all_star:
			output_file.write(line)
	output_file.close()


if __name__ == '__main__':
	# main('../origin_data/final_star2movie.txt', '../origin_data/all_star.txt', '../origin_data/final_star2movie1.txt')
	main('../origin_data/singer2song_with_comment.txt', '../origin_data/all_star.txt', '../origin_data/singer2song_with_comment1.txt')
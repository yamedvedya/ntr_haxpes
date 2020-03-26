#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request
import os
import requests
from bs4 import BeautifulSoup
from src.ntr_data_fitting.subfunctions import parse_sw


def get_sw_from_server(general_settings, structure, sample_folder=None):
    data = {
        # X-rays:
        'xway': str(general_settings['X_WAY']),
        'wave': str(general_settings['WAVE']),
        'line': general_settings['LINE'],
        'ipol': str(general_settings['IPOL']),

        # Substrate:
        'subway': str(general_settings['SUBWAY']),
        'code': general_settings['CODE'],
        'chem': general_settings['CODE'],
        'rho': str(general_settings['CODE']),
        'x0': general_settings['CODE'],
        'w0': str(general_settings['W0']),

        # Substrate surface:
        'sigma': str(general_settings['SIGMA']),
        'tr': str(general_settings['TR']),

        # Database Options for dispersion corrections df1, df2:
        'df1df2': str(general_settings['DF1DF2']),

        # X-rays:
        'scanmin': str(general_settings['SCANMIN']),
        'scanmax': str(general_settings['SCANMAX']),
        'unis': str(general_settings['UNIS']),
        'nscan': str(general_settings['NSCAN']),

        # X-rays:
        'swflag': str(general_settings['SWFLAG']),
        'swref': str(general_settings['SWREF']),
        'swmin': str(general_settings['SWMIN']),
        'swmax': str(general_settings['SWMAX']),
        'swpts': str(general_settings['SWPTS']),

        # job watching option:
        'watch': '1',

        # profile:
        'profile': '',
    }

    for layer in structure:
        data['profile'] += 't = {} '.format(layer['thick'])
        data['profile'] += 'code = {}'.format(layer['comp1'])

        if layer['comp2']:
            data['profile'] += ' x = {} code2 = {}'.format(layer['x1'], layer['comp2'])

        if layer['comp3']:
            data['profile'] += ' x2 = {} code3 = {}'.format(layer['x2'], layer['comp3'])

        if layer['sigma']:
            data['profile'] += ' sigma = {}'.format(layer['sigma'])

        if layer['w0']:
            data['profile'] += ' w0 = {}'.format(layer['w0'])

        if layer['x0']:
            data['profile'] += ' x0 = {}'.format(layer['x0'])

        data['profile'] += '\n'

    print('Requesting SW')
    response = BeautifulSoup(requests.post('https://x-server.gmca.aps.anl.gov/cgi/ter_form.pl', data=data).content,
                             'html.parser')

    task = response.find_all('p')[0].find_all('b')[1].get_text()

    finished = False
    while not finished:
        status = BeautifulSoup(requests.get('https://x-server.gmca.aps.anl.gov/cgi/wwwwatch.exe?jobname={}'.format(task)).content,
            'html.parser')

        test_status = status.find_all('p')[0].get_text().strip()
        finished = ('Processing, please wait...' not in test_status) and ('Progress:' not in test_status)

    if status.find_all('th') == []:
        display_link = None
        download_link = None
        for line in status.find_all('p'):
            if 'Display sw-grd file' in line.get_text():
                display_link = line.find('a').get('href')
            if 'Download zipped results' in line.get_text():
                download_link = line.find('a').get('href')

        if display_link is not None:
            sw_data = requests.get('https://x-server.gmca.aps.anl.gov/{}'.format(display_link)).content
        else:
            raise RuntimeError('Cannot find link to results')

        if not "sw_profiles" in os.listdir(sample_folder):
            os.mkdir(os.path.join(sample_folder, "sw_profiles"))

        if download_link:
            urllib.request.urlretrieve("https://x-server.gmca.aps.anl.gov/{}".format(download_link),
                                       os.path.join(os.path.join(sample_folder, "sw_profiles"), '{}.zip'.format(task)))
    else:
        raise RuntimeError(status.find('th').get_text())

    return parse_sw(sw_data.decode("utf-8").strip().split('\n'))

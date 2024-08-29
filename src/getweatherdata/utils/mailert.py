#  -*-coding: utf-8 -*-

# - # - # - # - # - # - # - # - # - # - # - #
#         E-Mail alert system for WU        #
#      Sends an e-mail and text message     #
#                if errors occur            #
#                                           #
#   Created by Sebastien Rougerie-Durocher  #
#        last edit : 21 March 2024          #
#             -----------------             #
#                 Contact                   #
#  sebastien.rougerie-durocher@irda.qc.ca   #
#   for  more info and specific details.    #
# - # - # - # - # - # - # - # - # - # - # - #

'''
##############################################
###           Important mentions           ###
##############################################
This is heavily inspired by Alexandre Leca Mailert python script.
Purpose of this code is to be attached to any script that runs on a iterative basis.

The alpha version of this code can be found with Mailert_Alpha_v1.py under the section "for email (free) ---"

the email package is already included with the python library. It cannot be pip installed.

This script utilizes an html format.

##############################################
###             Code description           ###
##############################################
The code works by utilizing the smtplib python module
smtplib uses smtp protocol client to send emails to any internet machine (see https://docs.python.org/3/library/smtplib.html)
This is the way it goes:
- Need to have an email account from which to send the emails (python itself doesn't send the emails)
- This account needs to have its password showned on the script (hence taking a placeholder email account)
- From this account, we log in, write a message, and then send it to another email address
- The placeholder account is currently a gmail account. This required an activation of the "less secured access" option
located within the gmail account that allows 3rd party users to manipulate the account and do stuff with it.

The email.mime package allows for better layouts
(see https://nitratine.net/blog/post/how-to-send-an-email-with-python/)

The error message is the traceback message itself. To get this message, the traceback module needs to be imported within
the working script. Although it is also okay if some other form of message is used as well.

##############################################
###                 Example                ###
##############################################
Typical function call should look like this :
import Mailert_Beta as ml
if len(mailerrors)>0:
    ml.email_alert(ml.alert_msg(mailerrors,script_name),"UserName@gmail.com")

Where,
- mailerrors is a dictionnary
  - Key should the station
  - First item should be the error message
  - Second item should be the time at which the message appeared
  - Example : mailerrors['Saint-Bruno'] = 'Traceback (most recent call last):... ' should be the format for the first item.

- script_name is the code of the script as a str

'''
# Imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging


# Functions
def alert_msg(mail, script_name):
    """
        This function gets the mailerrors dictionary and creates the alert message to
        be sent to the administrator after each download.
    """
    scrtipt_name_HTML = '<span style="color: #496dd0"> {} </span>'.format(script_name)  # Write script name in blue

    msg = """Hello,

    this is an automatic e-mail sent to inform that an error occurred in the
    workings of the {0} script: <br />""".format(scrtipt_name_HTML)
    for k in mail.keys():
        mail[k][0] = mail[k][0].replace('\n', '<br />')  # Make msg compatible with html format

        # Write the email itself
        byst = "<br /> - Station {0} ({1}) received the following error message : ".format(k, mail[k][1])
        error_msg = "<br /> <p><i> {} </i></p> <br />".format(mail[k][0])

        # Gather everything together
        msg = msg + byst + error_msg

    # Write the ending
    tail = """Please do not answer this e-mail.

    To contact the administrator, please send an inquiry to :
    sebastien.rougerie-durocher@irda.qc.ca
    <br /> <br />
    Thank you and have a nice day.
    """

    # Gather everything together
    msg = msg + tail
    return msg


def send_email_alert(mail_body, user_address="sebroug93@gmail.com"):
    '''
    This function sends an e-mail from a Gmail address to inform about a problem
    occurring on file download.

    :param mail_body: The content of the email, basically the error message, as established by the alert_msg function.
    :param user_address: By default, my own email address.
    :return: none
    '''

    gmail_user = 'seb.irda.emailsender@gmail.com'
    # gmail_password = 'ASDqwe!"/'
    gmail_password = "vkta ulwb sbeq lcol "

    expeditor = "sebrdurocher@gmail.com"
    message = MIMEMultipart()
    message['From'] = expeditor
    message['To'] = user_address
    message['Subject'] = "Code Problem Advisory"
    message.attach(MIMEText(mail_body, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(gmail_user, gmail_password)
        text = message.as_string()
        smtp_server.sendmail(gmail_user, user_address, text)
    print('Message sent successfully')


    # try:
    #     server = smtplib.SMTP_SSL('smtp.gmail.com', 465)  # code 465 for SSL, 587 for TLS
    #     server.ehlo()
    #     server.login(gmail_user, gmail_password)
    #
    #     server.close()
    # except BaseException as e:  # Return and print exception for no matter what is going on
    #     logging.error('Failed to send email: %s', e)




# #Test drive---------------------------------------------------------------------------------------------------------
# import datetime
# import traceback
# staname = 'Test_Station'
# script_name = 'Test_Script.py'#Outside the python console (= interactive interpreter) use : os.path.basename(__file__)
# mailerrors = {}
# x = 'a'
# try :
#    xx = 1 + x
#
# except BaseException:
#    Error_msg = traceback.format_exc()
#
#    mailerrors[staname] = [Error_msg, datetime.datetime.now().strftime("%Y/%m/%d at %H:%M:%S")]
#    # mailerrors[staname] = [str(e), datetime.datetime.utcnow().strftime("%Y/%m/%d at %H:%M:%S")]
#
# if len(mailerrors)>0:
#    send_email_alert(alert_msg(mailerrors, script_name))
#

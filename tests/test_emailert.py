import unittest
from unittest.mock import patch, MagicMock
from utils import mailert
import smtplib


# note : I didn't write this
class TestMailert(unittest.TestCase):

    @patch('smtplib.SMTP_SSL')
    def test_email_alert_sends_email_on_successful_login(self, mock_smtp):
        # Test check if send_email_alert sends an email on successful login
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value = mock_smtp_instance

        mail_body = "Test email body"
        user_address = "test@example.com"

        mailert.send_email_alert(mail_body, user_address)

        mock_smtp.assert_called_once_with('smtp.gmail.com', 465)
        mock_smtp_instance.login.assert_called_once_with('seb.irda.emailsender@gmail.com', 'vkta ulwb sbeq lcol ')
        mock_smtp_instance.sendmail.assert_called_once_with('seb.irda.emailsender@gmail.com', user_address, ANY)
        mock_smtp_instance.close.assert_called_once()

    @patch('smtplib.SMTP_SSL')
    def test_email_alert_logs_error_on_failed_login(self, mock_smtp):
        # test check if function correctly logs an error when login fails
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value = mock_smtp_instance
        mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(535, b'Error')

        with self.assertLogs('utils.mailert', level='ERROR') as cm:
            mailert.email_alert("Test email body", "test@example.com")

        self.assertIn('Failed to send email: SMTPAuthenticationError(535, b\'Error\')', cm.output)

    def test_alert_msg_formats_email_body_correctly(self):
        mail = {
            'Test_Station': ['Test error message', '2024/03/21 at 12:00:00']
        }
        script_name = 'Test_Script.py'

        expected_output = """Hello,

    this is an automatic e-mail sent to inform that an error occurred in the
    workings of the <span style="color: #496dd0"> Test_Script.py </span> script: <br /><br /> - Station Test_Station (2024/03/21 at 12:00:00) received the following error message : <br /> <p><i> Test error message </i></p> <br />Please do not answer this e-mail.

    To contact the administrator, please send an inquiry to :
    sebastien.rougerie-durocher@irda.qc.ca
    <br /> <br />
    Thank you and have a nice day.
    """

        self.assertEqual(mailert.alert_msg(mail, script_name), expected_output)

if __name__ == '__main__':
    unittest.main()

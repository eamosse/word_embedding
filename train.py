from openpyxl import load_workbook
def sendMail(_from,_to,_msg):
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(_msg)
    msg['Subject'] = '[COO] Note du partiel'
    server = smtplib.SMTP('smtp.gmail.com', 587)  # port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('eamosse@gmail.com', '!Edou30121983!')
    server.sendmail(_from,_to, msg.as_string())
    server.close()

wb = load_workbook(filename = 'workbook.xlsx')
sheet_ranges = wb['Notes']
max = 38
for i in range(3,38):
    note = float(sheet_ranges['F{}'.format(str(i))].value)
    msg = "Bonjour {}, \n".format(sheet_ranges['B{}'.format(str(i))].value)
    msg += "Votre note pour l'examen COO : {}/20.\n".format(note)
    msg += "N'hésitez pas à revenir vers moi si vous avez des questions.\n"
    msg+="\n\nCordialement, \nAmosse Edouard"
    sendMail("eamosse@gmail.com", sheet_ranges['E{}'.format(str(i))].value,msg)
    print("{} {}".format(sheet_ranges['E{}'.format(str(i))].value,note))
